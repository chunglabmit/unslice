import numpy as np
import zarr, os, time, sys, csv
from functools import partial 
import scipy.ndimage as ndi 
import pandas as pd
import skimage.transform 
from skimage.morphology import binary_dilation, selem 
from scipy.io import loadmat 
from scipy.spatial import cKDTree 
import multiprocessing as mp
from . import IO as io 
from itertools import product 
import torch 
import torch.nn.functional as F 
from tqdm import tqdm 
import warnings 
import contextlib 
 
# from precomputed_tif.client import DANDIArrayReader
if sys.platform.startswith("linux"):
    is_linux = True
    import tempfile
else:
    is_linux = False
    import mmap

def resample_image(img, current_resolution, desired_resolution, order=1):
    '''
    Resamples an image to be the desired size. 

    Inputs:
    img = 3D or 2D array of the image to be resampled 
    current_resolution = (x,y,z) or (x,y) micron/pixel resolution of original image 
    desired_resolution = (x,y,z) or (x,y) micron/pixel resolution of desired image 
    order = 0 for nearest, 1 for bilinear, and 3 for cubic interpolation 

    Outputs:
    img_resampled = resampled image array
    '''
    resample_factor = []
    for i in range(len(current_resolution)):
        resample_factor.append(current_resolution[i]/desired_resolution[i])
    resample_factor = tuple(resample_factor) 
    if resample_factor == (1,)*len(resample_factor):
        # Now we can convert from one zarr to another 
        img_resampled = img 
    else:
        warnings.filterwarnings('ignore', '.*output shape of zoom.*')
        img_resampled = ndi.zoom(img, resample_factor, order=order) 
    # img_resampled = skimage.transform.rescale(img, scale=resample_factor, order=order) 
    return img_resampled 

########### Resample data that's already in zarr format 
def resample_zarr(source_path, sink_path, resample_factor=(1,1,1), num_workers=24):
    source_zarr = zarr.open(source_path, mode='r')
    size = source_zarr.shape 
    chunks = source_zarr.chunks 
    rx,ry,rz = resample_factor 
    new_chunks = (int(np.round(chunks[0]*rx)) , int(np.round(chunks[1]*ry)), int(np.round(chunks[2]*rz)))
    new_img_size = (int(np.round(rx*size[0])),int(np.round(ry*size[1])),int(np.round(rz*size[2])))

    sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_path), shape=new_img_size, 
                            chunks=new_chunks, dtype='uint16', overwrite=True)

    new_coords = get_chunk_coords(new_img_size, new_chunks)
    old_coords = get_chunk_coords(size, chunks)

    coords = list(tuple(zip(old_coords, new_coords)))
    p = mp.Pool(num_workers)
    f = partial(_resample_chunk, source_zarr, sink_zarr, resample_factor)
    list(tqdm(p.imap(f, coords), total=len(coords)))
    p.close()
    p.join()

def _resample_chunk(source_zarr, sink_zarr, resample_factor, coords):
    old_coords, new_coords = coords 
    xr,yr,zr = old_coords
    xn,yn,zn = new_coords 
    img = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] 
    reimg = resample_image(img, resample_factor, (1,1,1))
    sink_zarr[xn[0]:xn[1],yn[0]:yn[1],zn[0]:zn[1]] = reimg


    
def upsample_zarr(source_path, sink_path, factor, final_image_size=None, 
                  num_workers=None, is_mask=True):
    '''
    Same as resample_zarr but for upsampling (have to deal with uneven resampling.
    
    Inputs:
    source_path - source zarr path 
    sink_path - sink zarr path 
    factor - int, number of times ot upsample 
    final_image_size - (optional), tuple of image size in case it's different from what we get from upsampling 
    num_workers - (optional), number of processes to run 
    is_mask - (optional), if True, assumes this is a mask and will return a binary mask. default: True 
    '''

     # Get the new shape of the image 
    source_zarr = zarr.open(source_path, mode='r')
    factor = [1/factor[i] for i in range(3)] # _resample_zarr_chunk takes a factor in opposite direction
    chunks = (*final_image_size[:2],source_zarr.chunks[2])
    resampled_zarr = zarr.zeros(shape=final_image_size, chunks=chunks, 
                                dtype=source_zarr.dtype, store=zarr.DirectoryStore(sink_path), overwrite=True)
    
    # Resample chunks 
    if num_workers is None or num_workers == 1:
        _resample_zarr_chunk([source_zarr, resampled_zarr, factor, is_mask], [[0,source_zarr.shape[i]] for i in range(3)])
    else:
        ## Multiprocessing version 
        p = mp.Pool(num_workers)
        f = partial(_resample_zarr_chunk, [source_zarr, resampled_zarr, factor, is_mask])
        coords_ranges = get_chunk_coords(source_zarr.shape, source_zarr.chunks)
        list(tqdm(p.imap(f, coords_ranges,chunksize=10),total=len(coords_ranges)))
        p.close()
        p.join()
        
        
def convert_to_zarr(source_path, sink_path, img_size, file_names=None, num_slices=1, chunks=None, 
                    num_workers=8, rotate=None, flip=None):
    '''
    Converts raw images into zarr file.
    
    Inputs:
    source_path - str, the path to the image. If this is a file, then we read directly the whole thing,
                  if it's a directory, read slice by slice the images. 
    sink_path - str, the path to the zarr file. 
    img_size - (x,y) size of the images 
    file_names - regexp str, regular expression of the filenames we want to read in (default: None) 
    num_slices - int, number of slices to load (assuming loading from slices)
    chunks - (x,y,z) size of the chunks to divide the zarr file into (default: None) 
    num_workers - parallel processes to use when converting 3D image to zarr (default: 8)
    rotate - int (can be 90, 180, or 270 clockwise), will rotate by the number of degrees specified slice by slice 
    flip - str ('x', 'y', or 'z'), the dimension in which to flip the image (x means horizontal flip, y means verticle flip)
    '''
    
    print('Starting...')
    if chunks is None:
        z1 = zarr.open(sink_path, mode='w', shape=img_size, dtype=np.uint16, overwrite=True) 
    else:
        z1 = zarr.open(sink_path, mode='w', shape=img_size, chunks=chunks, dtype=np.uint16, overwrite=True)
    print('Finished creating zarr')

    if os.path.isdir(source_path):
        if num_workers is not None and num_workers > 1:
            p = mp.Pool(num_workers)
            filenames = os.listdir(source_path)
            if num_slices == 1:
                name_ind_tuple = [(os.path.join(source_path,j),i) for i,j in enumerate(os.listdir(source_path))]
                f = partial(_convert_to_zarr_2d, z1)
                p.map(f, name_ind_tuple)

            else:
                filename = os.path.join(source_path, file_names) 
                print('Starting to convert')
                ## This is for slice-by-slice chunking 
                DataFileRange = [{'x':all,'y':all,'z':[i,i+num_slices]} for i in np.arange(0,len(filenames),num_slices)]
                f = partial(_convert_to_zarr_3d, [z1, filename]) 
                # p.map(f, DataFileRange)
                list(tqdm(p.imap(f, DataFileRange), total=len(DataFileRange)))
                
                ## Trying out different chunking and loading 
                # zranges = [[i,i+num_slices] for i in np.arange(0,len(filenames),num_slices)]
                # xranges = [[i,i+chunks[0]] for i in np.arange(0,img_size[0],chunks[0])]
                # yranges = [[i,i+chunks[1]] for i in np.arange(0,img_size[1],chunks[1])]
                # f = partial(_convert_to_zarr_3d, [z1, xranges, yranges, filename])
                # p.map(f, zranges)
            p.close()
            p.join() 
        else:
            for i,filename in enumerate(os.listdir(source_path)):
                _convert_to_zarr_2d(z1, (os.path.join(source_path,filename),i))
    elif os.path.isfile(source_path):
        img = io.readData(source_path)
        z1[:] = img 

        
    return z1 



## The following is for assigning whole slice chunks. 
def _convert_to_zarr_3d(args, DataFileRange):
    # args[0] = zarr_object (zarr file)
    # args[1] = filenames (regexp of the filenames of images in director) 
    zarr_object, filename = args 
    zarr_object[:,:,DataFileRange['z'][0]:DataFileRange['z'][1]] = io.readData(filename, **DataFileRange)
    print('Converted slices %d-%d'%(DataFileRange['z'][0],DataFileRange['z'][1]-1))
    
def _convert_to_zarr_2d(zarr_object, name_ind_tuple):
    '''
    Assigns a slice of the zarr array to a slice 
    
    Inputs:
    zarr_object - zarr array 
    name_ind_tuple - (str, int) tuple containing the name of the image and the slice index 
    '''
    
    name, ind = name_ind_tuple 
    zarr_object[:,:,ind] = io.readData(name) 
    print('Converted slice %d'%(ind))


################### convert from zarr to zarr, current way too slow to be of us     
def convert_zarr_chunks(source_zarr_path, sink_zarr_path,  new_chunks,sink_shape=None, num_workers=8):
    '''
    Converts zarr to zarr with different chunking.
    Can change the shape by adjusting sink_shape (if None, then it'll be the same)
    '''
    source = zarr.open(source_zarr_path, mode='r')
    if sink_shape is None:
        sink_shape = source.shape
    sink = zarr.open(sink_zarr_path, mode='w', shape=sink_shape,
                     chunks=new_chunks, dtype=source.dtype)#, synchronizer=zarr.ThreadSynchronizer())
    
    # Need to convert from source chunks to new chunks. we do this by loading in the max chunk size of both
    load_chunk = np.maximum(np.array(list(source.chunks)), np.array(list(sink.chunks)))
    chunk_coords = get_chunk_coords(source.shape, load_chunk)
    for chunk_coord in tqdm(chunk_coords):
        # Load in the big chunk first, so we can parallel process the small chunks
        xr,yr,zr = chunk_coord # global coordinates
        # print(chunk_coord) 
        large_chunk = source[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
        p = mp.Pool(num_workers)
        
        # Now we get the local chunk coords 
        small_chunk_coords = get_chunk_coords(large_chunk.shape, new_chunks)
        chunks = [(large_chunk[x[0]:x[1],y[0]:y[1],z[0]:z[1]], (x,y,z)) for (x,y,z) in small_chunk_coords]

        f = partial(_convert_zarr_to_zarr, sink, chunk_coord)
        #list(tqdm(p.imap(f, chunks), total=len(chunks)))
        p.map(f,chunks)
        p.close()
        p.join()
    
def _convert_zarr_to_zarr(sink_zarr, global_coord, chunks_and_coords):
    '''
    global_coord - [(),(),()], global chunk coordinates 
    chunks_and_coords - (chunks, local chunk coords): numpy ndarrays of all the chunk that needs to be assigned 
            i.e. the local chunk coordinates that correspond to chunk
    '''
    chunk, (xr,yr,zr) = chunks_and_coords
    # xr,yr,zr = chunk_coord
    xrl,yrl,zrl = global_coord # actual global coordinates

    #print('lol')
    start = time.time()
    # print(zr, sink_zarr.shape, source_zarr.shape)
    lim = [xrl[0]+xr[1],yrl[0]+yr[1],zrl[0]+zr[1]]
    for i,(ar,arl) in enumerate([(xr,xrl),(yr,yrl),(zr,zrl)]):
        if ar[1]-ar[0] > chunk.shape[i]:
            lim[i] = chunk.shape[i]+arl[0]+ar[0]

    # if zr[1] > source_array.shape[2]:
    #     zr[1] = source_array.shape[2]
    # if zrl[1] > sink_zarr.shape[2]:
    #     zrl[1] = sink_zarr.shape[2]
    
    # print(sink_zarr[xrl[0]+xr[0]:lim[0],
    #           yrl[0]+yr[0]:lim[1],
    #           zrl[0]+zr[0]:lim[2]].shape)
    sink_zarr[xrl[0]+xr[0]:lim[0],
              yrl[0]+yr[0]:lim[1],
              zrl[0]+zr[0]:lim[2]] = chunk
    #print("Converted x: %d-%d, y: %d-%d, z: %d-%d in %f seconds"%(xr[0],xr[1],yr[0],yr[1],zr[0],zr[1],time.time()-start))





############### Convert to zarr, crop, resample, flip, rotate ##############
def resample_rotate(img_input, resample_factor=(1,1,1), resample_order=1, rotate_angle=None, rotate_axes=(1,0), shm=None):
    if type(img_input) == tuple:
        # if tuple, then assume that we have a SharedMemory dtype 
        shm_input, idx = img_input 
        with shm_input.txn() as a: 
            img = a[:,:,idx[0]:idx[1]]
    else:
        img = img_input 
    if resample_factor != (1,1,1) and resample_factor is not None:
        img = resample_image(img, resample_factor, (1,1,1), order=resample_order)
    if rotate_angle is not None:
        img = rotate_img(img, rotate_angle, axes=rotate_axes)

    if shm is not None:
        with shm.txn() as b:
            b[:,:,idx[0]:idx[1]] = img 
        return 
    else:
        return img 


def rotate_img(img, angle, axes=(1,0)):
    return ndi.rotate(img, angle, axes=axes)


def _numpy_to_zarr(sink_zarr, array_and_coords, flip=(0,0,0)):
    '''
    Assigns numpy array to certain zarr coordinates, also allows for flips (reflections)

    Inputs:
    sink_zarr - zarr
    array_and_coords - tuple of the image (numpy array) and coordinates (list of lists)
                        coordinates: list [[x0,x1],[y0,y1],[z0,z1]] of coordinates to assign the npimg 
    flip - tuple of bools, determines which axes will be flipped. e.g. (0,1,1) means flip in Y and Z but not X
    '''
    img, coords = array_and_coords
    if type(img) == tuple:
        shm, local_coords = img 
        x,y,z = local_coords 
        with shm.txn() as a:
            img = a[x[0]:x[1],y[0]:y[1],z[0]:z[1]]
    xr,yr,zr = coords  
    if flip[0] == 1:
        img = np.flip(img, axis=0)
    if flip[1] == 1:
        img = np.flip(img, axis=1)
    if flip[2] == 1:
        img = np.flip(img, axis=2)

    try:
        sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = img.astype('uint16')
    except Exception as e:
        print(e)
        print(xr,yr,zr, img.shape)


def _assign_array(shm, axis, array_and_idx):
        idx,arr = array_and_idx 
        slc = [slice(None)] * len(arr.shape)
        slc[axis] = slice(idx*arr.shape[axis], (idx+1)*arr.shape[axis])
        with shm.txn() as a:
            a[tuple(slc)] = arr


def fast_numpy_concatenate(list_of_arrays, axis=-1, num_workers=8):
    '''
    Ultrast concatenation of arrays using parallelization
    '''
    if axis == -1:
        axis = list_of_arrays[0].shape[-1] 

    img_shape = list(list_of_arrays[0].shape)
    img_shape[axis] = len(list_of_arrays)*list_of_arrays[0].shape[axis]
    shm = SharedMemory(tuple(img_shape), list_of_arrays[0].dtype)

    
    args = [(i,arr) for (i,arr) in enumerate(list_of_arrays)]
    f = partial(_assign_array, shm, axis)

    p = mp.Pool(num_workers)
    p.map(f, args)
    p.close()
    p.join()

    with shm.txn() as a:
        final_array = np.array(a[:])
    return final_array 


def _temp_readData(filename, xrange, yrange, zrange, shm=None, z0=None, **resample_rotate_kwargs):
    '''
    Inputs:
    shm - if not None, then is SharedMemory array into which we assign the arrays
    **resample_rotate_kwargs: in case we want to do resampling and rotation as well 
    '''

    img = io.readData(filename, x=xrange, y=yrange, z=zrange)
    if len(resample_rotate_kwargs) > 0:
        img = resample_rotate(img, **resample_rotate_kwargs)
    if shm is not None:
        with shm.txn() as a:
            a[:,:,zrange[0]-z0:zrange[1]-z0] = img 
        return 
    else:
        return img 
    

def parallel_readData(fname, num_slices, num_workers, shm=None, resample_rotate=False,
                    resample_factor=(1,1,1), resample_order=1, rotate_angle=None, rotate_axes=(1,0), **DataFileRange):
    '''
    Speeding up io.readData using parallel processing.

    fname - str, the name of the files (can use regexp)
    num_slices - number of slices to read in one chunk
    num_workers - int, number of workers for parallel processing
    shm - SharedMemory, a SharedMemory array to store the array in. If None, this function returns a numpy array 
    resample_rotate - bool, if True, then perform resample_rotate as well 
    **DataFileRange - dictionary of 'x' 'y' 'z' coordinates to read 
    '''
    p = mp.Pool(num_workers)
    z0,z1 = DataFileRange['z']
    zslices = [[i,i+num_slices] for i in np.arange(z0,z1,num_slices)]
    zslices[-1][1] = z1 # to make sure we don't overshoot 

    if resample_rotate:
        f = partial(_temp_readData, fname, DataFileRange['x'], DataFileRange['y'], shm=shm, z0=z0,
                    resample_factor=resample_factor, resample_order=resample_order, rotate_angle=rotate_angle, rotate_axes=rotate_axes)
    else:
        f = partial(_temp_readData, fname, DataFileRange['x'], DataFileRange['y'], shm=shm, z0=z0)
    processed_chunks = list(tqdm(p.imap(f, zslices), total=len(zslices)))
    p.close()
    p.join()
    
    if shm is None:
        processed_large_chunk = fast_numpy_concatenate(processed_chunks, axis=2, num_workers=num_workers)
        return processed_large_chunk 
    else:
        return None 


    

def convert_to_zarr_v2(source_path, sink_paths, img_size, load_num_slices=None, resample_num_slices=1, file_names='img_[0-9]{4}.tif', chunks=None, 
                    num_workers=8, lateral_rotate_angle=None, flip=(0,0,0), crop_xcoords=None, crop_ycoords=None, crop_zcoords=None, 
                    resample_factor=(1,1)):
    '''
    Converts raw images into zarr file.
    
    Inputs:
    source_path - str, the path to the image. If this is a file, then we read directly the whole thing,
                  if it's a directory, read slice by slice the images. 
    sink_paths - str or tuple, the path to the zarr file. If tuple, then we convert into multiple zarrs (different resampling).  
    img_size - (x,y,z) size of the images 
    file_names - regexp str, regular expression of the filenames we want to read in (default: None) 
    load_num_slices - int, number of slices to load (assuming loading from slices)
    resample_num_slices - int, number of slices to resample at once 
    chunks - (x,y,z) size of the chunks to divide the zarr file into (default: None) 
    num_workers - parallel processes to use when converting 3D image to zarr (default: 8)
    lateral_rotate_angle - int, will rotate by the number of degrees specified laterally
    flip - tuple of bools, determines which axes will be flipped. e.g. (0,1,1) means flip in Y and Z but not X
    crop_xcoords - list [x0,x1], if None (default), then don't crop 
    crop_ycoords - list [y0,y1], if None (default), then don't crop
    crop_zcoords - list [z0,z1], if None (default), then don't crop
    resample_factor - tuple (x,y), no z because there is no good accurate way to downsample in this dimension in parallel
                     can also be tuple of tuples (if so, must match length of sink_path)
    '''
    warnings.filterwarnings("ignore") # ignore the fasttiff ij warning
    
    # First we need to compute the final size of the image 
    if crop_xcoords is not None:
        x0 = crop_xcoords[0]; x1 = crop_xcoords[1]
    else:
        x0 = 0; x1 = img_size[0]
    if crop_ycoords is not None:
        y0 = crop_ycoords[0]; y1 = crop_ycoords[1]
    else:
        y0=0; y1 = img_size[1]
    if crop_zcoords is not None:
        z0 = crop_zcoords[0]; z1 = crop_zcoords[1]
    else:
        z0 = 0; z1 = img_size[2] 
    xsize = x1-x0; ysize = y1-y0; zsize = z1-z0 
     
    if resample_factor is not None:
        rx,ry = resample_factor 
        xsize = int(np.round(xsize*rx))
        ysize = int(np.round(ysize*ry))

    if lateral_rotate_angle == 90 or lateral_rotate_angle == 270:
        xsize_copy = xsize
        xsize = ysize 
        ysize = xsize_copy  

    new_img_size = (xsize, ysize, zsize)

    if type(sink_paths) == str:
        sink_path = sink_paths 

    if chunks is None:
        sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_path), shape=new_img_size, dtype='uint16', overwrite=True)
    else:
        sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_path), shape=new_img_size, chunks=chunks, dtype='uint16', overwrite=True)
    
    # print(sink_zarr.shape)
    if os.path.isdir(source_path):
        if load_num_slices is None:
            load_num_slices = chunks[2] 

        filename = os.path.join(source_path, file_names) # regexp
        DataFileRange = [{'x':[x0,x1],'y':[y0,y1],'z':[i,i+load_num_slices]} for i in np.arange(z0,z1,load_num_slices)]
        DataFileRange[-1]['z'][1] = z1 # correct for this 
        
        # First alternative is loading each large chunk and parallelizing the operations within.
        # Second alternative is working in parallel for each large chunk 
        for i in range(len(DataFileRange)):
            print('Processing chunk x:%d-%d, y:%d-%d, z:%d-%d'%(x0,x1,y0,y1,DataFileRange[i]['z'][0],DataFileRange[i]['z'][1]))
            
            shm= SharedMemory((*new_img_size[:2],DataFileRange[i]['z'][1]-DataFileRange[i]['z'][0]), sink_zarr.dtype)
            # # Load in the big chunk first, so we can parallel process the small chunks
            # large_chunk = io.readData(filename, **DataFileRange[i])
            # Load in parallel 
            large_chunk = parallel_readData(filename, 1, num_workers, shm=shm, resample_rotate=True,
                                            resample_factor=resample_factor+(1,), resample_order=1, 
                                            rotate_angle=lateral_rotate_angle, rotate_axes=(1,0), **DataFileRange[i])
            print('Data I/O, resampling, rotation complete. Commencing flip and zarr assignment...')

            # Concatenate these in parallel 
            # processed_large_chunk = fast_numpy_concatenate(processed_chunks, axis=2, num_workers=num_workers)
            # processed_large_chunk = np.concatenate(tuple(processed_chunks), axis=2) 

            # Now we flip and convert to zarr 
            small_chunks = tuple(np.minimum(shm.shape,sink_zarr.chunks))
            small_chunk_coords = get_chunk_coords(shm.shape,small_chunks)
            global_coords = np.array(small_chunk_coords) # shape (num_chunks, 3, 2)
            global_coords[:,2,:] += i*load_num_slices # to compute the correct global coordinates 
            
            # Now transform the local coords to correspond to the global coordinates 
            small_chunk_coords = np.array(small_chunk_coords)
            if flip[0] == 1:
                small_chunk_coords[:,0,:] = np.flip(np.array([xsize-small_chunk_coords[:,0,:]]),2)   
            if flip[1] == 1:
                small_chunk_coords[:,1,:] = np.flip(np.array([ysize-small_chunk_coords[:,1,:]]),2)   
            if flip[2] == 1:
                nslices = shm.shape[2]
                small_chunk_coords[:,2,:] = np.flip(np.array([nslices-small_chunk_coords[:,2,:]]),2) 
                global_coords[:,2,:] = z1-z0-global_coords[:,2,:]
                global_coords[:,2,:] = np.flip(global_coords[:,2,:],1)

            # Get rid of negatives
            small_chunk_coords = small_chunk_coords * (small_chunk_coords>0)
            small_chunk_coords = list(small_chunk_coords) 
            global_coords = list(global_coords)

            # if type(processed_large_chunk) == np.ndarray:
            #     arrays = [processed_large_chunk[x[0]:x[1],y[0]:y[1],z[0]:z[1]] for (x,y,z) in small_chunk_coords]
            # else:
            #     arrays = [(shm_resample_rotate, (x,y,z)) for (x,y,z) in small_chunk_coords]

            arrays = [(shm, (x,y,z)) for (x,y,z) in small_chunk_coords]
            coords_list = [(x,y,z) for (x,y,z) in global_coords]
            arrays_and_coords = list(map(lambda q,r:(q,r), arrays, coords_list))

            p = mp.Pool(num_workers)
            f = partial(_numpy_to_zarr, sink_zarr, flip=flip)
            list(tqdm(p.imap(f, arrays_and_coords), total=len(arrays_and_coords)))
            p.close()
            p.join()

    elif os.path.isfile(source_path):
        # Add functionality to do rotation, flip, resampling, etc 
        print('Reading image')
        img = io.readData(source_path)
        sink_zarr[:] = img 

    return sink_zarr



########### Convert DANDI files to zarrs ########
# def dandi2zarr(urls, xrange, yrange, zrange, sink_zarr_path, resample_factor=(1,1,1), flip=(0,0,0), 
#                 chunks=3*(200,), num_workers=24):
#     '''
#     Converts DANDI files to zarrs
#     '''
    
#     dar = DANDIArrayReader(urls)

#     x0,x1 = xrange; xsize = x1-x0
#     y0,y1 = yrange; ysize = y1-y0
#     z0,z1 = zrange; zsize = z1-z0

#     if resample_factor is not None:
#         rx,ry,rz = resample_factor 
#         xsize = int(np.round(xsize*rx))
#         ysize = int(np.round(ysize*ry))
#         zsize = int(np.round(zsize*rz))
#         old_chunks = (int(np.round(chunks[0]/rx)), int(np.round(chunks[1]/ry)), int(np.round(chunks[2]/rz)))
#     else:
#         old_chunks = chunks 

#     old_img_size = (x1-x0,y1-y0,z1-z0)
#     new_img_size = (xsize, ysize, zsize)

#     sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_zarr_path), shape=new_img_size, chunks=chunks, dtype='uint16', overwrite=True)
    
    
#     zarr_coords = get_chunk_coords(new_img_size, chunks)
#     dandi_coords = np.array(get_chunk_coords(old_img_size, old_chunks))
#     dandi_coords[:,0] += x0 
#     dandi_coords[:,1] += y0
#     dandi_coords[:,2] += z0 
#     dandi_coords = list(dandi_coords)


#     coords = list(tuple(zip(dandi_coords, zarr_coords)))

#     p = mp.Pool(num_workers)
#     f = partial(_dandi2zarr, dar, sink_zarr, resample_factor=resample_factor)
#     list(tqdm(p.imap(f, coords), total=len(coords)))
#     p.close()
#     p.join()


# def _dandi2zarr(dar, sink_zarr, coords, resample_factor=(1,1,1), flip=(0,0,0)):
#     dandi_coords, zarr_coords = coords
#     xr,yr,zr = dandi_coords
#     xn,yn,zn = zarr_coords
#     img = dar.read_chunk(xr[0],xr[1],yr[0],yr[1],zr[0],zr[1])
#     img = np.swapaxes(img, 0, 2)
#     if resample_factor is not None:
#         img = resample_image(img,  resample_factor, (1,1,1), order=1)
#     sink_zarr[xn[0]:xn[1],yn[0]:yn[1],zn[0]:zn[1]] = img 





################## Allow multiple zarr creations in the same function call ########
def resample_rotate_v3(img, resample_factors=[(1,1,1)], resample_order=1, rotate_angle=None, rotate_axes=(1,0)):
    imgs = []
    for resample_factor in resample_factors:
        if resample_factor != (1,1,1) and resample_factor is not None:
            img = resample_image(img, resample_factor, (1,1,1), order=resample_order)
        if rotate_angle is not None:
            img = rotate_img(img, rotate_angle, axes=rotate_axes)
        imgs.append(img)
    return imgs 

def _temp_readData_v3(filename, xrange, yrange, zrange, shms=None, z0=None, **resample_rotate_kwargs):
    '''
    Inputs:
    shm - if not None, then is SharedMemory array into which we assign the arrays
    **resample_rotate_kwargs: in case we want to do resampling and rotation as well 
    '''
    img = io.readData(filename, x=xrange, y=yrange, z=zrange)
    if len(resample_rotate_kwargs) > 0:
        imgs = resample_rotate_v3(img, **resample_rotate_kwargs)
    if shms is not None:
        for idx,shm in enumerate(shms):
            print(type(shm), shm.shape)
            with shm.txn() as a:
                a[:,:,zrange[0]-z0:zrange[1]-z0] = imgs[idx]
        return None 
    else:
        return imgs 

def parallel_readData_v3(fname, num_slices, num_workers, shms=None, resample_rotate=False,
                    resample_factors=[(1,1,1)], resample_order=1, rotate_angle=None, rotate_axes=(1,0), **DataFileRange):
    '''
    Speeding up io.readData using parallel processing.

    fname - str, the name of the files (can use regexp)
    num_slices - number of slices to read in one chunk
    num_workers - int, number of workers for parallel processing
    shm - list of SharedMemory, a SharedMemory array to store the array in. If None, this function returns a numpy array 
    resample_rotate - bool, if True, then perform resample_rotate as well 
    **DataFileRange - dictionary of 'x' 'y' 'z' coordinates to read 
    '''
    p = mp.Pool(num_workers)
    z0,z1 = DataFileRange['z']
    zslices = [[i,i+num_slices] for i in np.arange(z0,z1,num_slices)]
    zslices[-1][1] = z1 

    if resample_rotate:
        f = partial(_temp_readData_v3, fname, DataFileRange['x'], DataFileRange['y'], shms=shms, z0=z0,
                    resample_factors=resample_factors, resample_order=resample_order, rotate_angle=rotate_angle, rotate_axes=rotate_axes)
    else:
        f = partial(_temp_readData_v3, fname, DataFileRange['x'], DataFileRange['y'], shms=shms, z0=z0)
    processed_chunks = list(tqdm(p.imap(f, zslices), total=len(zslices)))
    p.close()
    p.join()
    
    if shms is None:
        processed_chunks = map(list, zip(*processed_chunks))
        processed_large_chunks = []
        for idx,resample_factor in enumerate(resample_factors):
            processed_large_chunk = fast_numpy_concatenate(processed_chunks[idx], axis=2, num_workers=num_workers)
            processed_large_chunks.append(processed_large_chunk)
        return processed_large_chunks 

def convert_to_zarr_v3(source_path, sink_paths, img_size, load_num_slices=None, resample_num_slices=1, file_names='img_[0-9]{4}.tif', chunks=None, 
                    num_workers=8, lateral_rotate_angle=None, flip=(0,0,0), crop_xcoords=None, crop_ycoords=None, crop_zcoords=None, 
                    resample_factors=[(1,1)]):
    '''
    Converts raw images into zarr file, has the option to convert to multiple zarrs
    
    Inputs:
    source_path - str, the path to the image. If this is a file, then we read directly the whole thing,
                  if it's a directory, read slice by slice the images. 
    sink_paths - str or tuple, the path to the zarr file. If tuple, then we convert into multiple zarrs (different resampling).  
    img_size - (x,y,z) size of the images 
    file_names - regexp str, regular expression of the filenames we want to read in (default: None) 
    load_num_slices - int, number of slices to load (assuming loading from slices)
    resample_num_slices - int, number of slices to resample at once 
    chunks - (x,y,z) size of the chunks to divide the zarr file into (default: None) 
    num_workers - parallel processes to use when converting 3D image to zarr (default: 8)
    lateral_rotate_angle - int, will rotate by the number of degrees specified laterally
    flip - tuple of bools, determines which axes will be flipped. e.g. (0,1,1) means flip in Y and Z but not X
    crop_xcoords - list [x0,x1], if None (default), then don't crop 
    crop_ycoords - list [y0,y1], if None (default), then don't crop
    crop_zcoords - list [z0,z1], if None (default), then don't crop
    resample_factors - list of tuples (x,y), no z because there is no good accurate way to downsample in this dimension in parallel
                     can also be tuple of tuples (if so, must match length of sink_path)
    '''
    
    # First we need to compute the final size of the image 
    if crop_xcoords is not None:
        x0 = crop_xcoords[0]; x1 = crop_xcoords[1]
    else:
        x0 = 0; x1 = img_size[0]
    if crop_ycoords is not None:
        y0 = crop_ycoords[0]; y1 = crop_ycoords[1]
    else:
        y0=0; y1 = img_size[1]
    if crop_zcoords is not None:
        z0 = crop_zcoords[0]; z1 = crop_zcoords[1]
    else:
        z0 = 0; z1 = img_size[2] 
    xsize = x1-x0; ysize = y1-y0; zsize = z1-z0 
    
    if type(sink_paths) == str:
        sink_paths = [sink_paths]

    sink_zarrs = []
    new_img_sizes = []
    for (indx,sink_path) in enumerate(sink_paths):
        if resample_factors[indx] is not None:
            resample_factor = resample_factors[indx]
            rx,ry = resample_factor
            xsize = int(np.round(xsize*rx))
            ysize = int(np.round(ysize*ry))

        if lateral_rotate_angle == 90 or lateral_rotate_angle == 270:
            xsize_copy = xsize
            xsize = ysize 
            ysize = xsize_copy  

        new_img_size = (xsize, ysize, zsize)
        new_img_sizes.append(new_img_size)

        if chunks is None:
            sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_path), shape=new_img_size, dtype='uint16', overwrite=True)
        else:
            sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_path), shape=new_img_size, chunks=chunks, dtype='uint16', overwrite=True)
        sink_zarrs.append(sink_zarr)

    if os.path.isdir(source_path):
        if load_num_slices is None:
            load_num_slices = chunks[2] 

        filename = os.path.join(source_path, file_names)
        DataFileRange = [{'x':[x0,x1],'y':[y0,y1],'z':[i,i+load_num_slices]} for i in np.arange(z0,z1,load_num_slices)]
        DataFileRange[-1]['z'][1] = z1 
        
        # Load each large chunk and parallelize the operations within
        for i in range(len(DataFileRange)):
            print('Processing chunk x:%d-%d, y:%d-%d, z:%d-%d'%(x0,x1,y0,y1,DataFileRange[i]['z'][0],DataFileRange[i]['z'][1]))
            
            shms = []
            for indx,sink_zarr in enumerate(sink_zarrs):
                shm= SharedMemory((new_img_sizes[indx][:2],)+(DataFileRange[i]['z'][1]-DataFileRange[i]['z'][0]), sink_zarr.dtype)
                shms.append(shm)

            resample_factors = [resample_factors[kj]+(1,) for kj in range(len(resample_factors))]
            large_chunk = parallel_readData_v3(filename, 1, num_workers, shms=shms, resample_rotate=True,
                                            resample_factors=resample_factors, resample_order=1, 
                                            rotate_angle=lateral_rotate_angle, rotate_axes=(1,0), **DataFileRange[i])
            print('Data I/O, resampling, rotation complete. Commencing flip and zarr assignment...')

            # Now we flip and convert to zarr 
            for indx,shm in enumerate(shms):
                sink_zarr = sink_zarrs[indx]
                xsize,ysize,zsize = new_img_sizes[indx]

                small_chunks = tuple(np.minimum(shm.shape,sink_zarr.chunks))
                small_chunk_coords = get_chunk_coords(shm.shape,small_chunks)
                global_coords = np.array(small_chunk_coords) # shape (num_chunks, 3, 2)
                global_coords[:,2,:] += i*load_num_slices # to compute the correct global coordinates 
                
                # Now transform the local coords to correspond to the global coordinates 
                small_chunk_coords = np.array(small_chunk_coords)
                if flip[0] == 1:
                    small_chunk_coords[:,0,:] = np.flip(np.array([xsize-small_chunk_coords[:,0,:]]),2)   
                if flip[1] == 1:
                    small_chunk_coords[:,1,:] = np.flip(np.array([ysize-small_chunk_coords[:,1,:]]),2)   
                if flip[2] == 1:
                    nslices = shm.shape[2]
                    small_chunk_coords[:,2,:] = np.flip(np.array([nslices-small_chunk_coords[:,2,:]]),2) 
                    global_coords[:,2,:] = z1-z0-global_coords[:,2,:]
                    global_coords[:,2,:] = np.flip(global_coords[:,2,:],1)

                # Get rid of negatives
                small_chunk_coords = small_chunk_coords * (small_chunk_coords>0)
                small_chunk_coords = list(small_chunk_coords) 
                global_coords = list(global_coords)

                arrays = [(shm, (x,y,z)) for (x,y,z) in small_chunk_coords]
                coords_list = [(x,y,z) for (x,y,z) in global_coords]
                arrays_and_coords = list(map(lambda q,r:(q,r), arrays, coords_list))

                p = mp.Pool(num_workers)
                f = partial(_numpy_to_zarr, sink_zarr, flip=flip)
                list(tqdm(p.imap(f, arrays_and_coords), total=len(arrays_and_coords)))
                p.close()
                p.join()

    elif os.path.isfile(source_path):
        # Add functionality to do rotation, flip, resampling, etc 
        img = io.readData(source_path)
        sink_zarr[:] = img 



###################### Other utilities     
def mat2npy(matfilename, outfilename=None):
    '''
    Converts .mat file from matlab into numpy file.
    '''
    matdict = loadmat(matfilename)
    names = list(matdict.keys())
    for name in names:
        if name[:2] != '__':
            n = name 
            break 
    if outfilename is not None:
        np.save(outfilename, matdict[n])
    return np.asarray(matdict[n])


def mat2tif(matfilename, outfilename):
    '''
    Converts .mat file from matlab into TIF file. 
    '''
    matdict = loadmat(matfilename)
    names = list(matdict.keys())
    io.writeData(outfilename, matdict[names[-1]])
    
def saveroi(filepath, outpath, x, y, z):
    '''
    Saves an image in the desired range.
    '''
    DataFileRange = {'x': x, 'y': y, 'z': z}
    image = io.readData(filepath, **DataFileRange)
    io.writeData(outpath, image) 
    
def crop_img(img, xcoords=None, ycoords=None, zcoords=None):
    if xcoords is not None:
        x0,x1 = xcoords
    else:
        x0=0; x1=img.shape[0]
    if ycoords is not None:
        y0,y1 = ycoords 
    else:
        y0=0; y1=img.shape[1]
    if zcoords is not None:
        z0,z1 = zcoords
    elif len(img.shape) == 3:
        z0=0; z1=img.shape[2]

    if len(img.shape) == 2:
        return img[x0:x1,y0:y1]
    elif len(img.shape) == 3:
        return img[x0:x1,y0:y1,z0:z1]

def _crop_img_chunk(source_zarr, sink_zarr, source_coords):
    '''
    Serial kernel for cropping a zarr
    '''
    pass



def get_chunk_coords(image_shape, chunks):
    '''
    Utility to get the coordinates of chunks.
    Returns a list of list of lists 
    '''
    coords =  [[[i,i+chunks[0]],[j,j+chunks[1]],[k,k+chunks[2]]] \
                        for k in np.arange(0,image_shape[2],chunks[2]) \
                        for j in np.arange(0,image_shape[1],chunks[1]) \
                        for i in np.arange(0,image_shape[0],chunks[0])]
    c = np.array(coords)
    c[c[:,0,1]>image_shape[0],0,1] = image_shape[0]
    c[c[:,1,1]>image_shape[1],1,1] = image_shape[1]
    c[c[:,2,1]>image_shape[2],2,1] = image_shape[2]
    return c.tolist()
        
def chunk_dims(img_shape, chunk_shape):
    """Calculate the number of chunks needed for a given image shape

    Parameters
    ----------
    img_shape : tuple
        whole image shape
    chunk_shape : tuple
        individual chunk shape

    Returns
    -------
    nb_chunks : tuple
        a tuple containing the number of chunks in each dimension

    """
    return tuple(int(np.ceil(i/c)) for i, c in zip(img_shape, chunk_shape))      


class SharedMemory:
    """A class to share memory between processes

    Instantiate this class in the parent process and use in all processes.

    For all but Linux, we use the mmap module to get a buffer for Numpy
    to access through numpy.frombuffer. But in Linux, we use /dev/shm which
    has no file backing it and does not need to deal with maintaining a
    consistent view of itself on a disk.

    Typical use:

    shm = SharedMemory((100, 100, 100), np.float32)

    def do_something():

        with shm.txn() as a:

            a[...] = ...

    with multiprocessing.Pool() as pool:

        pool.apply_async(do_something, args)

    """

    if is_linux:
        def __init__(self, shape, dtype):
            """Initializer

            :param shape: the shape of the array

            :param dtype: the data type of the array
            """
            self.tempfile = tempfile.NamedTemporaryFile(
                prefix="proc_%d_" % os.getpid(),
                suffix=".shm",
                dir="/dev/shm",
                delete=True)
            self.pathname = self.tempfile.name
            self.shape = shape
            self.dtype = np.dtype(dtype)

        @contextlib.contextmanager
        def txn(self):
            """ A contextual wrapper of the shared memory

            :return: a view of the shared memory which has the shape and
            dtype given at construction
            """
            memory = np.memmap(self.pathname,
                               shape=self.shape,
                               dtype=self.dtype)
            yield memory
            del memory

        def __getstate__(self):
            return self.pathname, self.shape, self.dtype

        def __setstate__(self, args):
            self.pathname, self.shape, self.dtype = args

    else:
        def __init__(self, shape, dtype):
            """Initializer

            :param shape: the shape of the array

            :param dtype: the data type of the array
            """
            length = np.prod(shape) * dtype.itemsize
            self.mmap = mmap.mmap(-1, length)
            self.shape = shape
            self.dtype = dtype

        def txn(self):
            """ A contextual wrapper of the shared memory

            :return: a view of the shared memory which has the shape and
            dtype given at construction
            """
            memory = np.frombuffer(self.mmap, self.shape, self.dtype)
            yield memory
            del memory


def box_slice_idx(start, stop):
    """Creates an index tuple for a bounding box from `start` to `stop` using slices

    Parameters
    ----------
    start : array-like
        index of box start
    stop : array-like
        index of box stop (index not included in result)

    Returns
    -------
    idx : tuple
        index tuple for bounding box

    """
    return tuple(np.s_[a:b] for a, b in zip(start, stop))

def extract_box(arr, start, stop):
    """Indexes `arr` from `start` to `stop`

    Parameters
    ----------
    arr : array-like or SharedMemory
        input array to index
    start : array-like
        starting index of the slice
    stop : array-like
        ending index of the slice. The element at this index is not included.

    Returns
    -------
    box : ndarray
        resulting box from `arr`

    """
    idx = box_slice_idx(start, stop)
    if isinstance(arr, SharedMemory):
        with arr.txn() as a:
            box = a[idx]
    else:
        box = arr[idx]
    return box

                        
def fusepoints(points, intensities=None, distance=5):
    '''
    Uses KDTree to detect close points and fuse them. 
    '''
    tree = cKDTree(points)
    rows_to_fuse = tree.query_pairs(r=distance)

    for (r1, r2) in rows_to_fuse:
        points[r1] = (points[r1] + points[r2])//2

    duplicates = [r2 for (r1, r2) in rows_to_fuse]
    mask = np.ones(len(points), dtype=bool)
    mask[duplicates] = False
    
    if intensities is not None:
        return points[mask,:], intensities[mask,:]
    else:
        return points[mask,:]    
        
def accumarray(accmap, a, func=None, size=None, fill_value=0, dtype=None, num_workers=None):
    '''
    An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1) 
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array. 
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.
    num_workers : int, or None 
        If greater than 1 or not None, then will use parallel processing. 

    Returns
    -------
    out : ndarray
        The accumulated results.

        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.


    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accum(accmap, a)
    array([9, 7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([
            [[0,0],[0,0],[0,1]],
            [[0,0],[0,0],[0,1]],
            [[1,0],[1,0],[1,1]],
        ])
    >>> # Accumulate using a product.
    >>> accum(accmap, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    >>> # Same accmap, but create an array of lists of values.
    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)
    '''

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    if num_workers is None or num_workers <= 1:
        vals = np.empty(size, dtype='O')
        for s in product(*[range(k) for k in size]):
            vals[s] = []
        for s in product(*[range(k) for k in a.shape]):
            indx = tuple(accmap[s])
            val = a[s]
            vals[indx].append(val)

        # Create the output array.
        out = np.empty(size, dtype=dtype)
        for s in product(*[range(k) for k in size]):
            if vals[s] == []:
                out[s] = fill_value
            else:
                out[s] = func(vals[s])
        return out 
    else:
        p = mp.Pool(num_workers)
        f = partial(_accumarray_get_vals_serial,)
        vals = p.map(f, coord_ranges)

        f = partial(_accumarray_get_out_serial,fill_value)
        out = p.map(f, coord_ranges)
        p.close()
        p.join()

        return out 


def _accumarray_get_vals_serial():
    #

    pass

def _accumarray_get_out_serial(fill_value=0):
    #

    pass
 
def remove_from_array(all_points, some_points):
    '''
    Removes rows from an array.
    
    Inputs:
    all_points - array 
    some_points - array of points to remove 
    
    Outputs:
    new_array - array 
    idxs - array containing all of the relevant indices 
    '''
    idxs = np.argwhere(np.asarray([1-np.sum(np.prod(a==np.array(some_points),axis=1)) for a in all_points]) > 0)
    return all_points[idxs].squeeze(), idxs 
    


def intersect_2d(A,B):
    '''
    Similar to numpy's intersect_1d function but for 2D matrices when we want
    to compare the rows. Provides significant speedup over using sets. 
    
    Inputs:
    A,B - numpy ndarrays (2D) to be compared 
    
    Outputs:
    C - numpy ndarray (2D) containing the intersection of the two arrays.  
    
    '''
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C 
    
def union_2d(A,B):
    '''
    Similar to numpy's union1d function but for 2D matrices when we want to compare 
    the rows. 
    
    Inputs:
    A,B - numpy ndarrays (2D) to be compared 
    
    Outputs:
    C - numpy ndarray (2D) containing the union
    '''
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}

    C = np.union1d(A.view(dtype), B.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C 


def image_calculator(image1_path, image2_path, sink_path, 
                    image1_slice_range=None, mode='max', num_workers=1):
    '''
    Performs a certain operation between two 
    akin to ImageJ's image calculator function.

    Inputs:
    image1_path, image2_path - str, paths to the zarr arrays (image1 is the bottom image)
    sink_path - str, path to the final zarr 
    image1_slice_range - tuple, list, or array containing the slice range in image1 
                         for which to begin overlapping image2
    mode - str, the mode of image calculation. 
                possible options: 'max' (MIP), 'add', 'mean'
    num_workers - int, if greater than 1, then use multiprocessing. 
    
    Outputs:
    combined_img - Zarr array  containing result
    '''

    z1 = zarr.open(image1_path, mode='r')
    z2 = zarr.open(image2_path, mode='r')
    if image1_slice_range is None:
        # Use the whole image 
        image1_slice_range = [0,z1.shape[2]]
    b,t = image1_slice_range
    num_overlap = z1.shape[2]+z2.shape[2]-(t-b)
    #print(num_overlap)
    zr = zarr.open(sink_path, mode='w', shape=(*z1.shape[:2],num_overlap),
                    chunks=z1.chunks, dtype=z1.dtype)
    #print(zr.shape)

    # Initially assign the values of the images to the non-overlapping sections 
    if num_workers is None:
        start = time.time()
        if b > 0:
            zr[:,:,:b] = z1[:,:,:b]
        if t-b < z2.shape[2]:
            zr[:,:,t:] = z2[:,:,t-b:]
        c1 = z1[:,:,b:t]
        c2 = z2[:,:,:t-b]
        print("Time elapsed for memory-intensive assignment:",time.time()-start)


        if mode == 'max':
            overlap_region = np.maximum(c1,c2)
        elif mode == 'add':
            overlap_region = c1+c2 
        elif mode == 'mean':
            overlap_region = (c1+c2)//2 
        else:
            raise RuntimeError('That option is not available')
        zr[:,:,b:t] = overlap_region 
    else:
        # Assign the bottom section first 
        if b > 0:
            _assign_to_zarr(z1,[[0,z1.shape[0]],[0,z1.shape[1]],[0,b]],
                zr,[[0,zr.shape[0]],[0,zr.shape[1]],[0,b]], num_workers=8)
        # Now assign the top section
        if t-b < z2.shape[2]:
            _assign_to_zarr(z2,[[0,z2.shape[0]],[0,z2.shape[1]],[t-b,z2.shape[2]]],
                zr,[[0,zr.shape[0]],[0,zr.shape[1]],[t,zr.shape[2]]],num_workers=8)

        # now we do the image calculator 
        # coord_ranges = get_chunk_coords((*z1.shape[:2],t-b), z1.chunks)
        # p = mp.Pool(num_workers)
        # f = partial(_image_calculate_zarr,z1,z2,zr,image1_slice_range,mode)
        # list(tqdm(p.imap(f, coord_ranges), total=len(coord_ranges)))
        # p.close()
        # p.join()

        # For if the chunk sizes are too different 
        # Read a large chunk in and then do smaller chunk processing
        global_chunks_prelim = get_chunk_coords(z1.shape, (*z2.shape[:2],z1.chunks[2]))
        # First, get rid of all the chunks that are not relevant to the sections we care about
        global_chunks = []
        for ch in global_chunks_prelim:
            if ch[2][1] > b and ch[2][0] < t:
                global_chunks.append(ch)
        if b % z1.chunks[2] != 0:
            global_chunks[0][2][0] = b
        if global_chunks[-1][2][1] != t:
            global_chunks[-1][2][1] = t 

        for coords in global_chunks:
            start = time.time()
            xrl,yrl,zrl = coords
            print(xrl,yrl,zrl)

            print("Starting to read large image chunk")
            star = time.time()
            large_chunk = z2[xrl[0]:xrl[1],yrl[0]:yrl[1],zrl[0]-b:zrl[1]-b] # in the top image 
            print("finished large image chunk read in %f seconds"%(time.time()-star))

            #print(large_chunk.shape, z1.chunks)
            local_chunks = get_chunk_coords(large_chunk.shape, (*z1.chunks[:2],zrl[1]-zrl[0]))  
            img_chunks = [(large_chunk[c[0][0]:c[0][1],
                                          c[1][0]:c[1][1],
                                          c[2][0]:c[2][1]], c) for c in local_chunks]  
            if num_workers > 1:
                # Parallel version 
                p = mp.Pool(num_workers)
                # print(img_chunks[0][0].shape, local_chunks[0])
                f = partial(_image_calculate_zarr, z1,z2,zr,image1_slice_range,mode)
                list(tqdm(p.imap(f, img_chunks), total=len(img_chunks)))
                p.close()
                p.join()
            else:
                # Serial version 
                for chunk_pair in tqdm(img_chunks):
                    _image_calculate_zarr(z1,z2,zr,image1_slice_range,mode,chunk_pair)
    return zr 


#### Kernel and helper functions for image calculator 
def _image_calculate_zarr(zarr1,zarr2,zarrs,z1_slice_range,mode,coord_range):
    z1,z2=z1_slice_range
    if type(coord_range) == tuple:
        img_chunk, (xr,yr,zr) = coord_range
    else:
        xr,yr,zr = coord_range 
        img_chunk = zarr2[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
    if mode=='max':
        zarrs[xr[0]:xr[1],yr[0]:yr[1],zr[0]+z1:zr[1]+z1] =\
                                np.maximum(zarr1[xr[0]:xr[1],yr[0]:yr[1],zr[0]+z1:zr[1]+z1],
                                           img_chunk)
    elif mode=='add':
        zarrs[xr[0]:xr[1],yr[0]:yr[1],zr[0]+z1:zr[1]+z1] =\
                                 zarr1[xr[0]:xr[1],yr[0]:yr[1],zr[0]+z1:zr[1]+z1] +\
                                 img_chunk 
    elif mode=='mean':
        zarrs[xr[0]:xr[1],yr[0]:yr[1],zr[0]+z1:zr[1]+z1] =\
                                (zarr1[xr[0]:xr[1],yr[0]:yr[1],zr[0]+z1:zr[1]+z1] +\
                                 img_chunk) // 2
    else:
        raise RuntimeError("That option is not available")


def _assign_to_zarr(zarr1, zarr1_range, zarr_sink, zarr_sink_range, num_workers):
    '''
    Assign certain sections of zarr1 to certain sections of zarr_sink.
    More memory efficient than direct assignment.
    '''
    # Do this using multiprocessing 
    p = mp.Pool(num_workers)
    image_shape = (zarr1_range[0][1]-zarr1_range[0][0],
                        zarr1_range[1][1]-zarr1_range[1][0],
                        zarr1_range[2][1]-zarr1_range[2][0])
    # To make it efficient, we need to compute where we need to start with chunking
    # Check to see if we're starting on a chunk:
    if (zarr1_range[2][0] % zarr1.chunks[2]) != 0:
        overhang = zarr1.chunks[2] - (zarr1_range[2][0]%zarr1.chunks[2])
        hang_coords = get_chunk_coords((*zarr1.shape[:2],overhang),zarr1.chunks)
        # Now get coords for the rest of the image 
        coord_ranges = get_chunk_coords((*image_shape[:2],image_shape[2]-overhang),zarr1.chunks)
        coord_ranges = hang_coords + coord_ranges 
    else:
        coord_ranges = get_chunk_coords(image_shape, zarr1.chunks)
    
    f = partial(_assign_to_zarr_chunk, zarr1, zarr1_range, zarr_sink, zarr_sink_range)
    list(tqdm(p.imap(f,coord_ranges),total=len(coord_ranges)))
    p.close()
    p.join()



def _assign_to_zarr_chunk(zarr1, zarr1_range, zarr_sink, zarr_sink_range, coord_range):
    '''
    Serial kernel code for _assign_to_zarr.
    '''
    xr,yr,zr = coord_range
    # Update the actual global positions of these coordinates 
    xr1 = [xr[i] + zarr1_range[0][0] for i in range(2)]
    yr1 = [yr[i] + zarr1_range[1][0] for i in range(2)]
    zr1 = [zr[i] + zarr1_range[2][0] for i in range(2)]
    xrs = [xr[i] + zarr1_range[0][0] for i in range(2)]
    yrs = [yr[i] + zarr1_range[1][0] for i in range(2)]
    zrs = [zr[i] + zarr1_range[2][0] for i in range(2)]

    zarr_sink[xrs[0]:xrs[1],yrs[0]:yrs[1],zrs[0]:zrs[1]] = zarr1[xr1[0]:xr1[1],yr1[0]:yr1[1],zr1[0]:zr1[1]]





# function that adds exra slices to a zarr 
def concat_slices_zarr(source_zarr_path, sink_zarr_path):
    '''
    Concatenate zarr2 to zarr1. Only serial version currnetly available (multiprocessing version throws error)
    '''
    tmps = zarr.open(source_zarr_path, mode='r')
    extra_slice_zarr = zarr.zeros(shape=sizeI, chunks=tmps.chunks, 
                                   dtype=tmps.dtype, store=zarr.DirectoryStore(sink_zarr_path), overwrite=True)

    coords = get_chunk_coords(tmps.shape, tmps.chunks)
    for coord in tqdm(coords):
        # using multiprocessing throws an Input / Output error again 
        _convert_zarr_to_zarr(tmps, extra_slice_zarr, coord)


def points_to_image(file_name, output_name, coord_range):
    # file_name: assume that it's a .txt file with 3 columnbs 
    # coord_range: needed to subtract it when we put it in the actual image 
    
    if file_name[-3:] == 'txt' or file_name[-3:] == 'csv':
        points = np.loadtxt(file_name)
    elif file_name[-3:] == 'npy':
        points = np.load(file_name)


    xr, yr, zr = coord_range 
    # Make sure to only take the points within the range 
    points = points[(points[:,0]>=xr[0]) * (points[:,0]<xr[1])]
    points = points[(points[:,1]>=yr[0]) * (points[:,1]<yr[1])]
    points = points[(points[:,2]>=zr[0]) * (points[:,2]<zr[1])]
    points[:,0] = points[:,0] - xr[0]
    points[:,1] = points[:,1] - yr[0]
    points[:,2] = points[:,2] - zr[0]
    points = points.astype('int')

    img = np.zeros((xr[1]-xr[0],yr[1]-yr[0],zr[1]-zr[0]),dtype='bool')
    
    print("Image shape:",img.shape)
    img[points[:,0], points[:,1], points[:,2]] = True 
    
    io.writeData(output_name, img.astype('uint8'))


def pts_to_image(pts,size_image, radius=0):
    '''
    Takes a list of coordinates and creates an array (image) consisting of dots
    with radius "radius" at those points. Mostly used for overlaying detected / ground truth points
    with the original images.
    '''
    im = np.zeros(size_image,dtype='uint16')
    im[tuple(pts.T)] = 255
    if radius > 0:
        footprint = selem.disk(radius)
        imout= np.zeros(im.shape,dtype='uint16')
        for z in range(size_image[2]):
            imout[:,:,z] = binary_dilation(im[:,:,z],selem=footprint)
        return imout 
    else:
        return im 


########## Converting numpy to JSON (mostly for viewing in Nuggt) ####
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

def numpy_to_json(numpy_array, json_sink_path):
    with open(json_sink_path,'w') as write_file:
        json.dump(numpy_array, write_file, cls=NumpyArrayEncoder)
    # # Convert to csv first? 
    # np.savetxt('temp.csv',numpy_array,delimiter=",")
    # csvfile = open('temp.csv','r')
    # with open(json_sink_path, "w") as write_file:
    #     reader = csv.reader(csvfile,delimiter=',')
    #     for row in reader:
    #         json.dump(row,write_file)
    #         write_file.write('\n')

def read_annotations_json(json_path, name, sink_path=None):
    '''
    name is the name of the points annotations in the json (generated from neuroglancer annotations)
    '''

    # Opening JSON file
    f = open(json_path)
    data = json.load(f)
    
    for idx,dataset in enumerate(data['layers']):
        if dataset['name'] == name:
            indx = idx 
            break
    pts_list = data['layers'][indx]['annotations']
    pts = np.asarray(pd.DataFrame.from_dict(pts_list)['point'].to_list())
    f.close()

    if sink_path is not None:
        np.save(sink_path, pts)

    return pts 

    

# Function for getting indices of a given numpy array or array that correspond to another array
def get_indices(subset_array, big_array):
    '''
    big_array - nD numpy array that is the larger array from which we want to draw the indices
    subset_array - nD numpy array that is a subset of the larger array 
    '''
    ind_dict = dict((k,i) for i,k in enumerate(map(tuple,big_array)))
    inter = big_array.intersection(subset_array)
    indices = [ ind_dict[x] for x in inter ]
    indices.sort()
    return indices 



########## Functions for writing from zarr to tiffs #######

def convert_zarr_to_tiff(zarr_path, tiff_path, num_workers=24, zrange=None):
    if not os.path.isdir(tiff_path):
        os.makedirs(tiff_path)
    z = zarr.open(zarr_path,mode='r')
    if zrange is None:
        coords = get_chunk_coords(z.shape, (*z.shape[:2],z.chunks[2]))
    else:
        coords = np.asarray(get_chunk_coords((*z.shape[:2],zrange[1]-zrange[0]),(*z.shape[:2],z.chunks[2])))
        coords[:,2,:] += zrange[0]
        coords[coords[:,2,1]>z.shape[2],2,1] = z.shape[2]
        coords = coords.tolist()
    for coord in coords: # serially
        xr,yr,zr = coord
        if zr[1] > z.shape[2]:
            zr[1] = z.shape[2]
        print("Loading z %d - %d"%(zr[0],zr[1]))
        if num_workers is None:
            img_temp = z[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
            # print('Converting...')
            for i in tqdm(range(zr[1]-zr[0])):
                # print(i)
                _convert_to_tiff(tiff_path, img_temp, zr[0], i)
                #io.writeData(tiff_path + '/img_%04d.tif'%(zr[0]+i),img_temp[:,:,i])
        else:
            img_temp = SharedMemory((*z.shape[:2],zr[1]-zr[0]),z.dtype)
            parallel_readzarr(z, img_temp, coord, num_workers=num_workers)
            p = mp.Pool(num_workers)
            f = partial(_convert_to_tiff, tiff_path, img_temp, zr[0])
            list(tqdm(p.imap(f, np.arange(0,zr[1]-zr[0])),total=zr[1]-zr[0]))
            p.close()
            p.join()


def _convert_to_tiff(tiff_path, img, z, i):
    '''
    tiff_path: directory for converting to tiff
    img: numpy array of the image chunk being processed
    z: z index of the chunk (matters for the naming of tiffs)
    i: slice within the actualy image chunk
    '''
    if isinstance(img, SharedMemory):
        with img.txn() as a:
            io.writeData(tiff_path + '/img_%04d.tiff'%(z+i), a[:,:,i])
    else:
        io.writeData(tiff_path + '/img_%04d.tiff'%(z+i), img[:,:,i])
    # print("Converted %d"%i)




def write_zarr_to_tiff(source_zarr_path, coord_range, sink_tiff_path=None):
    '''
    write a tif file with given coordinates from zarr
    '''
    xr,yr,zr = coord_range 
    z = zarr.open(source_zarr_path, mode='r')
    img = z[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
    if sink_tiff_path is None:
        sink_tiff_path = source_zarr_path[:-5] + '_x%d-%d_y%d-%d_z%d-%d.tif'%(xr[0],xr[1],yr[0],yr[1],zr[0],zr[1])
    io.writeData(sink_tiff_path, img) 


def blockfs2zarr(img_path, channel_name, save_directory, xrange=None, yrange=None, zrange=None, chunks=None, num_workers=24):
    '''
    Convert a blockfs file (potentially crop) and save it as a zarr
    '''
    from blockfs import Directory
    channel = os.path.join(img_path, channel_name+'/precomputed.blockfs')
    DIRECTORY = Directory.open(channel)
    chunk_size = (DIRECTORY.x_block_size, DIRECTORY.y_block_size, DIRECTORY.z_block_size)

    old_img_size = DIRECTORY.x_extent, DIRECTORY.y_extent, DIRECTORY.z_extent
    if xrange is None:
        xrange = 0,old_img_size[0]
    if yrange is None:
        yrange = 0,old_img_size[1]
    if zrange is None:
        zrange = 0,old_img_size[2]

    new_img_size = xrange[1]-xrange[0], yrange[1]-yrange[0], zrange[1]-zrange[0]
    sink_zarr = zarr.create(store=zarr.DirectoryStore(save_directory), shape=new_img_size, 
                            chunks=chunks, dtype='uint16', overwrite=True)

    new_coords = get_chunk_coords(new_img_size, chunks) # chunks in the new zarr 
    coord_array = np.array(new_coords)
    coord_array[coord_array[:,0,1]>=new_img_size[0], 0, 1] = new_img_size[0]
    coord_array[coord_array[:,1,1]>=new_img_size[1], 1, 1] = new_img_size[1]
    coord_array[coord_array[:,2,1]>=new_img_size[2], 2, 1] = new_img_size[2]
    new_coords = coord_array.tolist()

    new_coord_array = np.array(new_coords)
    new_coord_array[:,0,:] += xrange[0]
    new_coord_array[:,1,:] += yrange[0]
    new_coord_array[:,2,:] += zrange[0]

    old_coords = new_coord_array.tolist()

    coords = list(tuple(zip(old_coords, new_coords)))
    p = mp.Pool(num_workers)
    f = partial(_blockfs2zarr, sink_zarr, DIRECTORY)
    list(tqdm(p.imap(f, coords), total=len(coords)))
    p.close()
    p.join()



def _blockfs2zarr(sink_zarr, blockfs_file, coords):
    '''
    kernel function for blockfs2zarr
    '''

    old_coords, new_coords = coords
    xro,yro,zro = old_coords # global coordinates in blockfs file (2becropped)
    xrn,yrn,zrn = new_coords # global coordinates in zarr file 
    
    # Need to get list of blocks to fit into this zarr 
    small_img_size = (xro[1]-xro[0],yro[1]-yro[0],zro[1]-zro[0]) # chunk size
    
    # Get blockfs chunk size
    bchunk_size = (blockfs_file.x_block_size, blockfs_file.y_block_size, blockfs_file.z_block_size)
    
    if (bchunk_size[0] < small_img_size[0]) or (bchunk_size[1] < small_img_size[1]) or (bchunk_size[2] < small_img_size[2]):
        # Get list of blockfs coordinates to be iterated through to assign to zarr
        internal_coords = get_chunk_coords(small_img_size,bchunk_size)
        tempp = np.array(internal_coords)
        tempp[tempp[:,0,1]>small_img_size[0],0,1] = small_img_size[0]
        tempp[tempp[:,1,1]>small_img_size[1],1,1] = small_img_size[1]
        tempp[tempp[:,2,1]>small_img_size[2],2,1] = small_img_size[2]
        
        tempp[:,0,:] += xro[0]
        tempp[:,1,:] += yro[0]
        tempp[:,2,:] += zro[0]
        internal_coords = tempp.tolist()
        
        img_temp = np.zeros(small_img_size, dtype='uint16')
        for internal_coord in internal_coords:
            xr,yr,zr = internal_coord
            try:
                b = blockfs_file.read_block(xr[0],yr[0],zr[0])
                img_temp[xr[0]-xro[0]:xr[1]-xro[0],yr[0]-yro[0]:yr[1]-yro[0],zr[0]-zro[0]:zr[1]-zro[0]] = np.swapaxes(b[:zr[1]-zr[0],:yr[1]-yr[0],:xr[1]-xr[0]],0,2)
            except Exception as e:
                print(xr[0],yr[0],zr[0])
                
        sink_zarr[xrn[0]:xrn[0]+img_temp.shape[0],yrn[0]:yrn[0]+img_temp.shape[1],zrn[0]:zrn[0]+img_temp.shape[2]] = img_temp 
    else:
        b = blockfs_file.read_block(xro[0],yro[0],zro[0])
        b_ = np.swapaxes(b[:zrn[1]-zrn[0],:yrn[1]-yrn[0],:xrn[1]-xrn[0]],0,2)
        sink_zarr[xrn[0]:xrn[1],yrn[0]:yrn[1],zrn[0]:zrn[1]] = b_ 


def zarr2blockfs():
    pass

def crop_zarr(source_zarr_path, sink_zarr_path, xrange=None, yrange=None, zrange=None, load_num_slices=None, num_workers=24):
    source_zarr = zarr.open(source_zarr_path, mode='r')
    chunks = source_zarr.chunks 
    size = source_zarr.shape 

    if xrange is None:
        xrange = [0,size[0]]
    if yrange is None:
        yrange = [0,size[1]]
    if zrange is None:
        zrange = [0,size[2]]

    new_img_size = (xrange[1]-xrange[0],yrange[1]-yrange[0],zrange[1]-zrange[0])
    sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_zarr_path), shape=new_img_size, 
                            chunks=chunks, dtype='uint16', overwrite=True)

    if load_num_slices is None:
        load_num_slices = chunks[2] 

    large_coords = get_chunk_coords(new_img_size,(*new_img_size[:2],load_num_slices))
    l = np.array(large_coords)
    l[:,0,:] += xrange[0]; l[:,1,:] += yrange[0]; l[:,2,:] += zrange[0]
    large_coords = l.tolist()


    for i in range(len(large_coords)):
        x0,x1 = large_coords[i][0]
        y0,y1 = large_coords[i][1]
        z0,z1 = large_coords[i][2]
        print('Processing chunk x:%d-%d, y:%d-%d, z:%d-%d'%(x0,x1,y0,y1,z0,z1))
        
        shm= SharedMemory((x1-x0,y1-y0,z1-z0), sink_zarr.dtype)
        parallel_readzarr(source_zarr, shm, large_coords[i],num_workers=num_workers)
 
        small_chunks = tuple(np.minimum(shm.shape,sink_zarr.chunks))
        small_chunk_coords = get_chunk_coords(shm.shape,small_chunks)
        global_coords = np.array(small_chunk_coords) # shape (num_chunks, 3, 2)
        global_coords[:,2,:] += i*load_num_slices # to compute the correct global coordinates 
        
        # Now transform the local coords to correspond to the global coordinates 
        small_chunk_coords = np.array(small_chunk_coords)

        # Get rid of negatives
        small_chunk_coords = small_chunk_coords * (small_chunk_coords>0)
        small_chunk_coords = list(small_chunk_coords) 
        global_coords = list(global_coords)


        arrays = [(shm, (x,y,z)) for (x,y,z) in small_chunk_coords]
        coords_list = [(x,y,z) for (x,y,z) in global_coords]
        arrays_and_coords = list(map(lambda q,r:(q,r), arrays, coords_list))

        p = mp.Pool(num_workers)
        f = partial(_numpy_to_zarr, sink_zarr, flip=(0,0,0))
        list(tqdm(p.imap(f, arrays_and_coords), total=len(arrays_and_coords)))
        p.close()
        p.join()

def parallel_readzarr(source_zarr, shm, coords, num_workers=24):
    chunks = source_zarr.chunks
    xr,yr,zr = coords
    new_img_sizea = (xr[1]-xr[0],yr[1]-yr[0],zr[1]-zr[0])
    chunk_coords = get_chunk_coords(new_img_sizea, chunks)

    global_chunk_coords = np.array(chunk_coords)
    global_chunk_coords[:,0,:] += xr[0]
    global_chunk_coords[:,1,:] += yr[0]
    global_chunk_coords[:,2,:] += zr[0]
    global_chunk_coords[global_chunk_coords[:,0,1]>xr[1],0,1] = xr[1]
    global_chunk_coords[global_chunk_coords[:,1,1]>yr[1],1,1] = yr[1]
    global_chunk_coords[global_chunk_coords[:,2,1]>zr[1],2,1] = zr[1]
    global_chunk_coords = global_chunk_coords.tolist()

    arrays = [(shm, (x,y,z)) for (x,y,z) in chunk_coords]
    coords_list = [(x,y,z) for (x,y,z) in global_chunk_coords]
    arrays_and_coords = list(map(lambda q,r:(q,r), arrays, coords_list))

    p = mp.Pool(num_workers)
    f = partial(_readzarr, source_zarr)
    list(tqdm(p.imap(f, arrays_and_coords), total=len(arrays_and_coords)))
    p.close()
    p.join()

def _readzarr(source_zarr, arrays_and_coords):
    arrays, global_coords = arrays_and_coords 
    xg,yg,zg = global_coords
    shm,small_coords = arrays 
    xs,ys,zs = small_coords
    # print(small_coords, global_coords)
    with shm.txn() as a:
        a[xs[0]:xs[1],ys[0]:ys[1],zs[0]:zs[1]] = source_zarr[xg[0]:xg[1],yg[0]:yg[1],zg[0]:zg[1]]


###########################

def resume_convert_to_zarr(source_path, sink_path, pc2_img_size, load_num_slices=None, resample_num_slices=1, file_names='img_[0-9]{4}.tif', chunks=None, 
                    num_workers=8, lateral_rotate_angle=None, flip=(0,0,0), crop_xcoords=None, crop_ycoords=None, crop_zcoords=None, 
                    resample_factor=(1,1)):
    sink_zarr = zarr.open(sink_path)
    filename = os.path.join(source_path, file_names)

    # Read in tiffs
    if crop_zcoords is None:
        zrange = [0,pc2_img_size[2]]
    else:
        zrange = crop_zcoords 

    zrange = crop_zcoords 
    for z in np.arange(zrange[0],zrange[1],load_num_slices):
        if z+load_num_slices > zrange[1]:
            zend = zrange[1]
        else:
            zend = z+load_num_slices
        print('Processing chunk z:%d-%d'%(z,zend))
        shm = SharedMemory((*pc2_img_size[:2],zend-z), 'uint16')
        # print(shm.shape)
        # DataFileRange = {'x':[0,pc2_img_size[0]],'y':[0,pc2_img_size[1]],'z':[z,zend]}
        DataFileRange = {'x':all,'y':all,'z':[z,zend]}
        # print(DataFileRange)
        large_chunk = parallel_readData(filename, 1, num_workers, shm=shm, resample_rotate=True,
                                        resample_factor=resample_factor+(1,), resample_order=1, 
                                        rotate_angle=lateral_rotate_angle, rotate_axes=(1,0), **DataFileRange)
        print('Data I/O, resampling, rotation complete. Commencing flip and zarr assignment...')

        small_chunks = tuple(np.minimum(shm.shape,sink_zarr.chunks))
        small_chunk_coords = get_chunk_coords(shm.shape,small_chunks)
        global_coords = np.array(small_chunk_coords) # shape (num_chunks, 3, 2)
        global_coords[:,2,:] += z # to compute the correct global coordinates 

        # Now transform the local coords to correspond to the global coordinates 
        small_chunk_coords = np.array(small_chunk_coords)

        # Get rid of negatives
        small_chunk_coords = small_chunk_coords * (small_chunk_coords>0)
        small_chunk_coords = list(small_chunk_coords) 
        global_coords = list(global_coords)

        arrays = [(shm, (x,y,z)) for (x,y,z) in small_chunk_coords]
        coords_list = [(x,y,z) for (x,y,z) in global_coords]
        arrays_and_coords = list(map(lambda q,r:(q,r), arrays, coords_list))

        p = mp.Pool(num_workers)
        f = partial(_numpy_to_zarr, sink_zarr, flip=flip)
        list(tqdm(p.imap(f, arrays_and_coords), total=len(arrays_and_coords)))
        p.close()
        p.join()