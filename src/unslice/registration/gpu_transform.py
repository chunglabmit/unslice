import multiprocessing
# from torch import multiprocessing 
import numpy as np
import zarr
from zarr import Blosc
from .. import utils 
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import product 
from .transform import * 


local_indices_default = None
moving_img_ram = None

def chunk_coordinates(shape, chunks):
    """Calculate the global coordaintes for each chunk's starting position

    Parameters
    ----------
    shape : tuple
        shape of the image to chunk
    chunks : tuple
        shape of each chunk

    Returns
    -------
    start : list
        the starting indices of each chunk

    """
    nb_chunks = utils.chunk_dims(shape, chunks)
    start = []
    for indices in product(*tuple(range(n) for n in nb_chunks)):
        start.append(tuple(i*c for i, c in zip(indices, chunks)))
    return start


def interpolate(image, coordinates, order=3):
    """Interpolate an image at a list of coordinates

    Parameters
    ----------
    image : array
        array to interpolate
    coordinates : array
        2D array (N, D) of N coordinates to be interpolated
    order : int
        polynomial order of the interpolation (default: 3, cubic)

    """
    output = map_coordinates(image,
                             coordinates.T,
                             output=None,
                             order=order,
                             mode='constant',
                             cval=0.0,
                             prefilter=True)
    return output


class TorchGridSampler:
    # Samples a grid in PyTorch 
    def __init__(self, values):
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.values = torch.from_numpy(values).float().cuda()
        else:
            self.values = torch.from_numpy(values).float()

    def __call__(self, grid):
        # if self.use_gpu:
        #     grid = torch.from_numpy(grid).float().cuda()
        # else:
        #     grid = torch.from_numpy(grid).float()

        values = F.grid_sample(self.values, grid, align_corners=True)  # A tensor
        return values

        # if self.use_gpu:
        #     return values.cpu().numpy()
        # else:
        #     return values.numpy()


def fit_grid_sampler(values):
    if len(values) == 2:
        interp_z = TorchGridSampler(values[0])
        interp_y = TorchGridSampler(values[1])
        return interp_z, interp_y
    elif len(values) == 3:
        interp_z = TorchGridSampler(values[0])
        interp_y = TorchGridSampler(values[1])
        interp_x = TorchGridSampler(values[2])
        return interp_z, interp_y, interp_x
        


def sample_grid(grid, interp, moving_shape, padding=2):
    interp_z, interp_y, interp_x = interp
    values_z = interp_z(grid)[0]  # (1, *chunks) tensor
    values_y = interp_y(grid)[0]
    values_x = interp_x(grid)[0]

    z_min = np.floor(values_z.min().cpu().numpy() - padding).astype(np.int)
    z_max = np.ceil(values_z.max().cpu().numpy() + padding).astype(np.int)
    y_min = np.floor(values_y.min().cpu().numpy() - padding).astype(np.int)
    y_max = np.ceil(values_y.max().cpu().numpy() + padding).astype(np.int)
    x_min = np.floor(values_x.min().cpu().numpy() - padding).astype(np.int)
    x_max = np.ceil(values_x.max().cpu().numpy() + padding).astype(np.int)
    transformed_start = np.array([z_min, y_min, x_min])
    transformed_stop = np.array([z_max, y_max, x_max])

    if np.any(transformed_stop < 0) or np.any(np.greater_equal(transformed_start, moving_shape)):
        return None
    else:
        start = np.array([max(0, s) for s in transformed_start])
        stop = np.array([min(e, s) for e, s in zip(moving_shape, transformed_stop)])
        if np.any(np.less_equal(stop - 1, start)):
            return None
        else:
            new_grid_x = (values_x - float(start[2])) / float(stop[2] - start[2] - 1) * 2 - 1
            new_grid_y = (values_y - float(start[1])) / float(stop[1] - start[1] - 1) * 2 - 1
            new_grid_z = (values_z - float(start[0])) / float(stop[0] - start[0] - 1) * 2 - 1
            new_grid = torch.stack([new_grid_x, new_grid_y, new_grid_z], dim=4)

            return start, stop, new_grid




def register_chunk(moving_img, fixed_img, output_img, values, start, chunks, padding=2, zrange=None):
    global local_indices_default
    global moving_img_ram
    # Get dimensions
    chunks = np.asarray(chunks)
    img_shape = np.asarray(output_img.shape)

    # Find the appropriate global stop coordinate and chunk shape accounting for boundary cases
    stop = np.minimum(start + chunks, img_shape)
    chunk_shape = np.array([b - a for a, b in zip(start, stop)])

    # Get z, y, x indices for each pixel
    if np.all(chunks == chunk_shape):
        local_indices = local_indices_default  
    else:
        local_indices = np.indices(chunk_shape)  # pretty slow so only do this when we have to
    global_indices = np.empty_like(local_indices)  # faster than zeros_like
    for i in range(global_indices.shape[0]):
        global_indices[i] = local_indices[i] + start[i]
    global_indices = torch.from_numpy(global_indices).float().cuda()

    # Make the grid with normalized coordinates [-1, 1]
    grid_y = (global_indices[1] / float(img_shape[1] - 1)) * 2 - 1
    grid_z = (global_indices[0] / float(img_shape[0] - 1)) * 2 - 1
    if len(global_indices) == 3:
        grid_x = (global_indices[2] / float(img_shape[2] - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y, grid_z], dim=3).unsqueeze(0)
    else:
        grid = torch.stack([grid_y, grid_z], dim=2).unsqueeze(0)

    # Sample the transformation grid
    interp = fit_grid_sampler(values)
    if len(global_indices) == 3:
        result = sample_grid(grid, interp, moving_img.shape, padding)
    else: 
        result = sample_grid_2d(grid, interp, moving_img.shape, padding)
    if result is not None:
        moving_start, moving_stop, moving_grid = result
        # Get the chunk of moving data
        if zrange is not None:
            if (moving_start[2] >= zrange[0]) and (moving_stop[2] <= zrange[1]):
                moving_start = (*moving_start[:2],moving_start[2]-zrange[0])
                moving_stop = (*moving_stop[:2],moving_stop[2]-zrange[0])
                moving_data = utils.extract_box(moving_img_ram, moving_start, moving_stop)

                if not np.any(moving_data):
                    interp_chunk = np.zeros(chunk_shape, output_img.dtype)
                else:
                    # interpolate the moving data
                    moving_data = moving_data.reshape((1, 1, *moving_data.shape)).astype(np.float)
                    moving_data_tensor = torch.from_numpy(moving_data).float().cuda()
                    interp_chunk = F.grid_sample(moving_data_tensor, moving_grid, align_corners=True).cpu().numpy()[0, 0]
            else:
                return 
        else:
            moving_data = utils.extract_box(moving_img_ram, moving_start, moving_stop)

            if not np.any(moving_data):
                interp_chunk = np.zeros(chunk_shape, output_img.dtype)
            else:
                # interpolate the moving data
                moving_data = moving_data.reshape((1, 1, *moving_data.shape)).astype(np.float)
                moving_data_tensor = torch.from_numpy(moving_data).float().cuda()
                interp_chunk = F.grid_sample(moving_data_tensor, moving_grid, align_corners=True).cpu().numpy()[0, 0]

    else:
        interp_chunk = np.zeros(chunk_shape, output_img.dtype)


    # write results to disk
    if len(global_indices) == 3:
        output_img[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = interp_chunk
    else:
        output_img[start[0]:stop[0], start[1]:stop[1]] = interp_chunk


def _register_chunk(args):
    register_chunk(*args)

def _assign_zarr_to_shm(zarr1, num_workers, z0):
    global moving_img_ram # this is the RAM zarr object 
    # Do this using multiprocessing 
    p = mp.Pool(num_workers)
    coord_ranges = np.asarray(utils.get_chunk_coords(moving_img_ram.shape, zarr1.chunks))
    coord_ranges[:,2,:] += z0
    coord_ranges[coord_ranges[:,2,1]>zarr1.shape[2],2,1] = zarr1.shape[2]
    coord_ranges = list(coord_ranges)

    f = partial(_assign_zarr_to_shm_chunk, zarr1, z0)
    list(tqdm(p.imap(f,coord_ranges),total=len(coord_ranges)))
    p.close()
    p.join()



def _assign_zarr_to_shm_chunk(zarr_, z0, coord_range):
    global moving_img_ram 
    xr,yr,zr = coord_range
    # moving_img_ram[xr[0]:xr[1],yr[0]:yr[1],zr[0]-z0:zr[1]-z0] = zarr_[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
    with moving_img_ram.txn() as aa:
        aa[xr[0]:xr[1],yr[0]:yr[1],zr[0]-z0:zr[1]-z0] = zarr_[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]


def register(moving_img, fixed_img, output_img, values, chunks, nb_workers, padding=2):
    global local_indices_default
    global moving_img_ram
    max_memory = 700 # in GB, probably should be like 90% of your computer's RAM 
    # multiprocessing.set_start_method('spawn', force=True)
    
    # Cache local indices
    local_indices_default = np.indices(chunks)

    # Compute number of bytes if loaded into memory, assuming uint16 datatype
    moving_img_memory = np.prod(moving_img.shape)*2/1e9 
    print("Moving image size:",moving_img_memory,"GB")
    if moving_img_memory < max_memory:
        # Load moving img

        # Option A: fastest option: for images smaller than RAM
        moving_img_ram = zarr.create(shape=moving_img.shape,
                                     chunks=moving_img.chunks,
                                     dtype=moving_img.dtype,
                                     compressor=Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE))
        moving_img_ram[:] = moving_img

        start_coords = chunk_coordinates(output_img.shape, chunks)
        if nb_workers <= 1 or nb_workers is None:
            for i, start_coord in tqdm(enumerate(start_coords), total=len(start_coords)):
                start = np.asarray(start_coord)
                args = (moving_img, fixed_img, output_img, values, start, chunks, padding)
                register_chunk(*args)
        else:
        ## Multiprocessing option 
            args_list = [(moving_img, fixed_img, output_img, values, 
                          np.array(start_coord), chunks, padding) for start_coord in start_coords]
            with multiprocessing.Pool(nb_workers) as pool:
                # pool.starmap(register_chunk, args_list)
                list(tqdm(pool.imap(_register_chunk, args_list), total=len(args_list)))
    elif True:
        ## This option makes it slow but is scalable to any size image 
        moving_img_ram = moving_img 

        start_coords = chunk_coordinates(output_img.shape, chunks)
        if nb_workers <= 1 or nb_workers is None:
            for i, start_coord in tqdm(enumerate(start_coords), total=len(start_coords)):
                start = np.asarray(start_coord)
                args = (moving_img, fixed_img, output_img, values, start, chunks, padding)
                register_chunk(*args)
        else:
        ## Multiprocessing option 
            args_list = [(moving_img, fixed_img, output_img, values, 
                          np.array(start_coord), chunks, padding) for start_coord in start_coords]
            with multiprocessing.Pool(nb_workers) as pool:
                # pool.starmap(register_chunk, args_list)
                list(tqdm(pool.imap(_register_chunk, args_list), total=len(args_list)))

    else:
        ## Option D: make "big chunks" and load these into memory, and then parallelize within each 
        # Load in full x,y, and 2*chunks in z, but have one z chunk worth of overlap
        all_zs = np.arange(0,moving_img.shape[2],moving_img.chunks[2])

        for zcoord in all_zs:
            if zcoord+2*moving_img.chunks[2]>moving_img.shape[2]:
                zend = moving_img.shape[2] 
            else:
                zend = zcoord+2*moving_img.chunks[2]
            zrange = (zcoord,zend)

            print("Warping z=%d to %d"%(zcoord,zend))
            # moving_img_ram = zarr.create(shape=(*moving_img.shape[:2],zend-zcoord),
            #                              chunks=moving_img.chunks,
            #                              dtype=moving_img.dtype,
            #                              compressor=Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE))
            # moving_img_ram[:] = moving_img[:,:,zcoord:zend]
            moving_img_ram = utils.SharedMemory((*moving_img.shape[:2],zend-zcoord),'uint16')
            _assign_zarr_to_shm(moving_img, 24, zcoord)

            start_coords = chunk_coordinates(output_img.shape, chunks)

            args_list = [(moving_img, fixed_img, output_img, values, 
                          np.array(start_coord), chunks, padding, zrange) for start_coord in start_coords]
            with multiprocessing.Pool(nb_workers) as pool:
                # pool.starmap(register_chunk, args_list)
                list(tqdm(pool.imap(_register_chunk, args_list), total=len(args_list)))







############################# Obsolete functions (or for 2D) ###############################
def register_slice(moving_img, zslice, output_shape, values, fixed_shape, padding=2):
    """Apply transformation and interpolate for a single z slice in the output

    Parameters
    ----------
    moving_img : zarr array
        input image to be interpolated
    zslice : int
        index of the z-slice to compute
    output_shape : tuple
        size of the 2D output
    values : ndarray
        grid for the nonlinear registration
    fixed_shape : tuple
        shape of the fixed image in 3D, used for interpolating grid
    padding : int, optional
        amount of padding to use when extracting pixels for interpolation

    Returns
    -------
    registered_img : ndarray
        registered slice from the moving image

    """
    # Get dimensions
    img_shape = np.asarray((1, *output_shape))  # (z, y, x)
    local_indices = np.indices(img_shape)  # (3, z, y, x), first all zeros
    global_indices = local_indices
    global_indices[0] = zslice

    # Make the grid with normalized coordinates [-1, 1]
    # The last axis need to be in x, y, z order... but the others are in z, y, x
    grid = np.zeros((1, 1, *fixed_shape[1:], 3))  # (1, z, y, x, 3)
    grid[..., 0] = 2 * global_indices[2] / (fixed_shape[2] - 1) - 1
    grid[..., 1] = 2 * global_indices[1] / (fixed_shape[1] - 1) - 1
    grid[..., 2] = 2 * global_indices[0] / (fixed_shape[0] - 1) - 1

    # Sample the transformation grid
    interp = fit_grid_sampler(values)
    moving_coords, transformed_start, transformed_stop = sample_grid_coords(grid, interp, padding)

    # Read in the available portion data (not indexing outside the moving image boundary)
    moving_start = np.array([max(0, s) for s in transformed_start])
    moving_stop = np.array([min(e, s) for e, s in zip(moving_img.shape, transformed_stop)])
    moving_data = utils.extract_box(moving_img, moving_start, moving_stop)  # decompresses data from disk

    # interpolate the moving data
    moving_coords_local = moving_coords - np.array(moving_start)
    interp_values = interpolate(moving_data, moving_coords_local, order=1)
    registered_img = np.reshape(interp_values, output_shape)
    return registered_img

def sample_grid_coords(grid, interp, padding=2):
    interp_z, interp_y, interp_x = interp
    values_z = interp_z(grid)[0]  # (1, *chunk_shape)
    values_y = interp_y(grid)[0]
    values_x = interp_x(grid)[0]
    coords = np.column_stack([values_z.ravel(), values_y.ravel(), values_x.ravel()])

    # Calculate the moving chunk bounding box
    z_tensor = torch.from_numpy(values_z).float().cuda()
    y_tensor = torch.from_numpy(values_y).float().cuda()
    x_tensor = torch.from_numpy(values_x).float().cuda()
    z_min = np.floor(z_tensor.min().cpu().numpy() - padding).astype(np.int)
    z_max = np.ceil(z_tensor.max().cpu().numpy() + padding).astype(np.int)
    y_min = np.floor(y_tensor.min().cpu().numpy() - padding).astype(np.int)
    y_max = np.ceil(y_tensor.max().cpu().numpy() + padding).astype(np.int)
    x_min = np.floor(x_tensor.min().cpu().numpy() - padding).astype(np.int)
    x_max = np.ceil(x_tensor.max().cpu().numpy() + padding).astype(np.int)

    start = np.array([z_min, y_min, x_min])
    stop = np.array([z_max, y_max, x_max])

    return coords, start, stop

def sample_grid_coords_2d(grid, interp, padding=2):
    interp_y, interp_x = interp
    values_y = interp_y(grid)[0]
    values_x = interp_x(grid)[0]
    coords = np.column_stack([values_y.ravel(), values_x.ravel()])

    # Calculate the moving chunk bounding box
    y_tensor = torch.from_numpy(values_y).float().cuda()
    x_tensor = torch.from_numpy(values_x).float().cuda()
    y_min = np.floor(y_tensor.min().cpu().numpy() - padding).astype(np.int)
    y_max = np.ceil(y_tensor.max().cpu().numpy() + padding).astype(np.int)
    x_min = np.floor(x_tensor.min().cpu().numpy() - padding).astype(np.int)
    x_max = np.ceil(x_tensor.max().cpu().numpy() + padding).astype(np.int)

    start = np.array([y_min, x_min])
    stop = np.array([y_max, x_max])

    return coords, start, stop

def sample_grid_2d(grid, interp, moving_shape, padding=2):
    interp_y, interp_x = interp
    values_y = interp_y(grid)[0]
    values_x = interp_x(grid)[0]

    y_min = np.floor(values_y.min().cpu().numpy() - padding).astype(np.int)
    y_max = np.ceil(values_y.max().cpu().numpy() + padding).astype(np.int)
    x_min = np.floor(values_x.min().cpu().numpy() - padding).astype(np.int)
    x_max = np.ceil(values_x.max().cpu().numpy() + padding).astype(np.int)
    transformed_start = np.array([y_min, x_min])
    transformed_stop = np.array([y_max, x_max])

    if np.any(transformed_stop < 0) or np.any(np.greater_equal(transformed_start, moving_shape)):
        return None
    else:
        start = np.array([max(0, s) for s in transformed_start])
        stop = np.array([min(e, s) for e, s in zip(moving_shape, transformed_stop)])
        if np.any(np.less_equal(stop - 1, start)):
            return None
        else:
            new_grid_x = (values_x - float(start[1])) / float(stop[1] - start[1] - 1) * 2 - 1
            new_grid_y = (values_y - float(start[0])) / float(stop[0] - start[0] - 1) * 2 - 1
            new_grid = torch.stack([new_grid_x, new_grid_y], dim=3)

            return start, stop, new_grid


