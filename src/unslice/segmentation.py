# Module with segmentation tools (and surface finding tools) for HUBRIS 

import numpy as np
from sklearn.cluster import KMeans 
import cv2 
import maxflow 
import time 
import zarr
from tqdm import tqdm 
import multiprocessing as mp 
from maxflow import fastmin 
from functools import partial 
import matplotlib.pyplot as plt 
from .utils import get_chunk_coords, read_annotations_json, pts_to_image
from skimage.filters import threshold_otsu, gaussian
from skimage import exposure, morphology
import scipy.ndimage as ndi
from PIL import Image, ImageDraw
#from mclahe import mclahe 
import hubris.IO as io 

from numcodecs import Blosc, Zstd

# The temp path   
temp_zarr_path = 'temp/sat_temp.zarr'
USE_NUGGT = True # if we want to use neuroglancer to visualize 
    
def kernel_graphcut(image, **opt):
    '''
    Implementation of a kernel graph cut for multi-region segmentation as described in:
    
    M.B. Salah - "Multiregion Image Segmentation by Parametric Kernel Graph Cuts", 
    IEEE Transactions on Image Processing, Vol. 20, No. 2, 2011
    
    Inputs:
    image - 2D or 3D array (if 3D, will just perform slice-by-slice) 
    
    **opt:
    saturate_image_threshold - float [0,1], fraction of pixels to saturate in the image (default: 0)
    k - int, number of segmentation classes (default: 2)
    alpha - float, smoothness parameter where the larger means a more smooth segmentation (default: 1)
    num_workers - int, number of workers for parallel computation (default: 8) 
  
    
    Outputs:
    segmented_image - 2D or 3D array containing labels for each pixel  
    '''
    
    if 'saturate_image_threshold' in opt:
        saturate_image_threshold = opt['saturate_image_threshold']
    else:
        saturate_image_threshold = 0 
    if 'k' in opt:
        k = opt['k']
    else:
        k = 2
    if 'alpha' in opt:
        alpha = opt['alpha']
    else:
        alpha = 1
    if 'min_threshold' in opt:
        min_threshold = opt['min_threshold']
    else:
        min_threhsold = 0 
    if 'num_workers' in opt:
        num_workers = opt['num_workers']
    else:
        num_workers = 8
        
    # # Saturate the image 
    # if saturate_image_threshold != 0:
    #     if saturate_image_threshold > 1:
    #         raise RuntimeError("Please enter a valid value between 0 and 1 for saturate_image_threshold.")
    #     else:
    #         sat_image = _saturate_image(image, saturate_image_threshold) 
    if saturate_image_threshold > 0:
        sat_image = _saturate_image(image, saturate_image_threshold, min_threshold)
    else:
        sat_image = image 

    if len(sat_image.shape) == 3:
        # Serial version 
        if num_workers <= 1 or num_workers is None: 
            segmented = np.zeros(sat_image.shape, dtype='uint8')
            for slic in range(sat_image.shape[2]):
                im = sat_image[:,:,slic]
                segmented[:,:,slic] = _kgc_kernel(alpha, k, im) 
        else:
            img_list = []
            for slic in range(image.shape[2]):
                img_list.append(sat_image[:,:,slic])
            f = partial(_kgc_kernel, alpha, k) 
            p = mp.Pool(num_workers)
            segmented = p.map(f, img_list) 
            segmented = np.swapaxes(np.swapaxes(np.asarray(segmented),0,2),0,1)
            
    elif len(sat_image.shape) == 2:
        segmented = _kgc_kernel(alpha, k, sat_image)
    else:
        raise RuntimeError("Input image should be 2- or 3-D. (Found %d-D array)"%len(sat_image.shape))
    
    return segmented 
    
def _kgc_kernel(alpha, k, im):
    '''
    Kernel for computing the Kernel Graph cut 
    '''
    kernel_graph_cut = False # whether to do normal graph cut or kernel graph cut 
    
    
    include_while_loop = False  
    ite = 0
    num_iters = 5 # number of iterations 
    energy_global_min = 1e8

    if kernel_graph_cut:
        # Initialization: cluster the data into k regions 
        print('Start kmeans')
        start = time.time()
        data = im.reshape(-1,1)
        kmeans = KMeans(n_clusters=k, random_state=42, n_jobs=1).fit(data)
        c = kmeans.cluster_centers_
        
        Dc = np.zeros(im.shape+(k,))
        
        if include_while_loop: 
            while ite < num_iters:
                ite += 1
                
                for ci in range(k):
                    Dc[:,:,ci] = 1-_kernel_rbf(im, c[ci])

                # Smoothness
                Sc = alpha*np.ones((k,k)) - alpha*np.eye(k)
                new_labels = fastmin.abswap_grid(Dc,Sc)
                energy = fastmin.energy_of_grid_labeling(Dc, Sc, new_labels)
                
                for q in range(k-1):
                    P1 = np.argwhere(new_labels == 1)
                    K = _kernel_rbf(im[P1[:,0],P1[:,1]], c[q+1])
                    smKI = np.sum(im[P1[:,0],P1[:,1]] * K)
                    smK = np.sum(K)
                    if len(P1) != 0:
                        c[q+1] = smKI/smK
                    else:
                        c[q+1] = 1e9
                
                
                if energy <= energy_global_min:
                    energy_global_min = energy 
                    L_global_min = new_labels.copy()
                    c_global_min = c.copy()
                    
                c = c[c != 1e9]
            
            new_labels = L_global_min 
            
                
        else: # Don't use the while loop 
            for ci in range(k):
                Dc[:,:,ci] = 1-_kernel_rbf(im, c[ci])

            # Smoothness
            Sc = alpha*np.ones((k,k)) - alpha*np.eye(k)
            new_labels = fastmin.abswap_grid(Dc,Sc)
    else:
        g = maxflow.Graph[float](k)
        nodeids = g.add_grid_nodes(im.shape)
        structure = np.array([[1,1,1],[1,0,1],[1,1,1]]) # 8-connectivity 
        ## For 3D 
        # structure = np.ones((3,3,3))
        # structure[1,1,1] = 0 
        g.add_grid_edges(nodeids, weights=alpha, structure=structure, symmetric=True)
        g.add_grid_tedges(nodeids, im, im.max()-im)
        g.maxflow()
        sgm = g.get_grid_segments(nodeids)
        new_labels = np.int_(np.logical_not(sgm))
    
    print("Done with chunk")

    return new_labels
    
##### The following are for processing zarrs instead of the full numpy arrays 
    
def zarr_kernel_graphcut(source_path, sink_path, **opt):
    '''
    Same as kernel_graphcut that takes zarr as input instead of numpy array.
    
    We take advantage of the zarr file in order to use _saturate_image such that it doesn't 
    take up a bunch of memory. 
    
    
    Extra option:
    min_threhsold - float, pixel intensity value under which we set to 0 in the image 
    sample_slices - bool, if True, we do a sample of 5 slices to see if it satisfies the user before processing. 
    '''
    global temp_zarr_path 
    
    if 'saturate_image_threshold' in opt:
        saturate_image_threshold = opt['saturate_image_threshold']
    else:
        saturate_image_threshold = 0     
    if 'k' in opt:
        k = opt['k']
    else:
        k = 2
    if 'alpha' in opt:
        alpha = opt['alpha']
    else:
        alpha = 1
    if 'num_workers' in opt:
        num_workers = opt['num_workers']
    else:
        num_workers = 8
    if 'min_threshold' in opt:
        min_threshold = opt['min_threshold']
    else:
        min_image_threshold = 0 
    if 'sample_slices' in opt:
        sample_slices = opt['sample_slices']
    else:
        sample_slices = False 
        
    source_zarr = zarr.open(source_path, mode='r')
    sink_zarr = zarr.open(sink_path, mode='w', shape=source_zarr.shape, chunks=source_zarr.chunks, dtype=np.uint8)
    
    start = time.time()
    # print("Starting image preprocessing..")
    if saturate_image_threshold > 0:
        _saturate_image(source_zarr, saturate_image_threshold, min_threshold=min_threshold, num_workers=num_workers) 
        new_source_zarr = zarr.open(temp_zarr_path, mode='r')
    else:
        new_source_zarr = source_zarr
    # print("Saturated image processed in %f minutes"%((time.time()-start)/60))
    
    
    
    # Allows users to see if they like what they see before continuing 
    # We sample near the beginning in order to see what the segmentation would look like  for the surface
    if sample_slices:
        num_samples = 6 
        # total_num_slices = new_source_zarr.shape[2]
        total_num_slices = 700 
        slice_nums = [(total_num_slices-1)//(num_samples-1)*i for i in range(num_samples)]
        for i in slice_nums:
            _kgc_kernel_zarr(alpha, k, new_source_zarr, sink_zarr, i) 
            f, (ax1, ax2, ax3) = plt.subplots(3,1)
            ax1.imshow(source_zarr[:,:,i], 'gray') 
            ax2.imshow(new_source_zarr[:,:,i], 'gray')
            ax3.imshow(sink_zarr[:,:,i], 'gray')
            plt.show() 
            
        check = input("Continue with these parameters? ([y]/n)")
        if check == 'n' or check == 'N':
            return 
        
    
    f = partial(_kgc_kernel_zarr, alpha, k, new_source_zarr, sink_zarr) 
    p = mp.Pool(num_workers)
    p.map(f, np.arange(source_zarr.shape[2])) 
    p.close()
    p.join() 

def _kgc_kernel_zarr(alpha, k, source_zarr, sink_zarr, slicenum):
    '''
    Kernel for computing the Kernel Graph cut but for zarrs 
    '''
    kernel_graph_cut = False # whether to do normal graph cut or kernel graph cut 
    
    im = source_zarr[:,:,slicenum] 
    
    include_while_loop = False  
    ite = 0
    num_iters = 5 # number of iterations 
    energy_global_min = 1e8

    if kernel_graph_cut:
        # Initialization: cluster the data into k regions 
        print('Start kmeans')
        start = time.time()
        data = im.reshape(-1,1)
        kmeans = KMeans(n_clusters=k, random_state=42, n_jobs=1).fit(data)
        c = kmeans.cluster_centers_
        
        Dc = np.zeros(im.shape+(k,))
        
        if include_while_loop: 
            while ite < num_iters:
                ite += 1
                
                for ci in range(k):
                    Dc[:,:,ci] = 1-_kernel_rbf(im, c[ci])

                # Smoothness
                Sc = alpha*np.ones((k,k)) - alpha*np.eye(k)
                new_labels = fastmin.abswap_grid(Dc,Sc)
                energy = fastmin.energy_of_grid_labeling(Dc, Sc, new_labels)
                
                for q in range(k-1):
                    P1 = np.argwhere(new_labels == 1)
                    K = _kernel_rbf(im[P1[:,0],P1[:,1]], c[q+1])
                    smKI = np.sum(im[P1[:,0],P1[:,1]] * K)
                    smK = np.sum(K)
                    if len(P1) != 0:
                        c[q+1] = smKI/smK
                    else:
                        c[q+1] = 1e9
                
                
                if energy <= energy_global_min:
                    energy_global_min = energy 
                    L_global_min = new_labels.copy()
                    c_global_min = c.copy()
                    
                c = c[c != 1e9]
            
            new_labels = L_global_min 
            
                
        else: # Don't use the while loop 
            for ci in range(k):
                Dc[:,:,ci] = 1-_kernel_rbf(im, c[ci])

            # Smoothness
            Sc = alpha*np.ones((k,k)) - alpha*np.eye(k)
            new_labels = fastmin.abswap_grid(Dc,Sc)
    else:
        g = maxflow.Graph[float](k)
        nodeids = g.add_grid_nodes(im.shape)
        structure = np.array([[1,1,1],[1,0,1],[1,1,1]]) # 8-connectivity 
        g.add_grid_edges(nodeids, weights=alpha, structure=structure, symmetric=True)
        g.add_grid_tedges(nodeids, im, im.max()-im)
        g.maxflow()
        sgm = g.get_grid_segments(nodeids)
        new_labels = np.int_(np.logical_not(sgm))
    sink_zarr[:,:,slicenum] = new_labels.astype('uint8')
    
    print("Done with slice %d"%slicenum)

    
def _kernel_rbf(img, value):
    
    # RBF-kernel parameters
    a = 2
    sigma = 0.5 
    
    K = np.exp(-(img-value)**a/sigma**2)
    
    return K 

############# 3D grpah cut ############################################################

def apply_morphological_op(img, operation, selem, radius):
    '''
    Apply morphological operation to a numpy array. 
    Inputs:

    img:        numpy ndarray, 3d array segmentation on which to apply morphological operations 
    radius:         Int: size fo the structuring element
    operation:         Possible options for morphological: None (no post-processing)
                                ('dilate', .., ..) morpholgoical dilation
                                ('erode', .., ..) morphological erosion 
                                ('close',.., ..) morphological closing
                                ('open',.., ..) morphological oepning 
    selem:         Possible options for structuring element:
                                (..,'ball',..)
                                (..,'cube',..)
                                (..,'octahedron',..)
    '''

    if selem == 'ball':
        s = morphology.ball(radius)
    elif selem == 'octahedron':
        s = morphology.octahedron(radius)
    elif selem == 'cube':
        s = morphology.cube(radius) 

    if operation == 'dilate':
        new_labels = morphology.binary_dilation(img, s)
    elif operation == 'erode':
        new_labels = morphology.binary_erosion(img, s)
    elif operation == 'close':
        new_labels = morphology.binary_closing(img, s)
    elif operation == 'open':
        new_labels = morphology.binary_opening(img, s)

    return new_labels

def graphcut3d(im, alpha, k):
    g = maxflow.Graph[float](k)
    nodeids = g.add_grid_nodes(im.shape)

    structure = np.ones((3,3,3)) # 26-connectivity 
    structure[1,1,1] = 0

    g.add_grid_edges(nodeids, weights=alpha, structure=structure, symmetric=True)
    g.add_grid_tedges(nodeids, im, im.max()-im)
    g.maxflow()

    sgm = g.get_grid_segments(nodeids)
    new_labels = np.int_(np.logical_not(sgm))

    new_labels = new_labels.astype('uint8')

    return new_labels


def _graphcut_kernel_zarr(alpha, k, source_zarr, sink_zarr, overlap, morphopts, coord_range):
    '''
    Kernel for computing the Kernel Graph cut but for zarrs 

    Inputs:
    
    alpha: int, smoothness weight for the graph
    k: int, number of classes for segmentation 
    source_zarr: zarr, zarr of the image to be segmented
    sink_zarr: zarr, zarr of the output segmentation
    overlap: int, number of pixels in each direction to overlap calculations
    morphopts: if not None:  
            tuple(str, str, int), option for post-processing the graph cut segmentation.
             str1: option for morphological operation
             str2: structuring element type  
             Int: size fo the structuring element
             See apply_morphological_op for more descriptions.
    coord_range: list of list of ints, a list of the coordinate chunk ranges for parallel processing  
    '''
    
    xr,yr,zr = coord_range 
    if overlap > 0:
        overlaps = np.asarray([[overlap]*2]*3)
        if xr[0] < overlap:
            overlaps[0][0] = xr[0]
        if xr[1] > source_zarr.shape[0] - overlap:
            overlaps[0][1] = source_zarr.shape[0]-xr[1]
        if yr[0] < overlap:
            overlaps[1][0] = yr[0]
        if yr[1] > source_zarr.shape[1] - overlap:
            overlaps[1][1] = source_zarr.shape[1]-yr[1]
        if zr[0] < overlap:
            overlaps[2][0] = zr[0]
        if zr[1] > source_zarr.shape[2] - overlap:
            overlaps[2][1] = source_zarr.shape[2]-zr[1]
        im = source_zarr[xr[0]-overlaps[0][0]:xr[1]+overlaps[0][1],
                          yr[0]-overlaps[1][0]:yr[1]+overlaps[1][1],
                          zr[0]-overlaps[2][0]:zr[1]+overlaps[2][1]] 
    else:
        im = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]

    if np.any(im):
        new_labels = graphcut3d(im, k, alpha)

        if morphopts is not None:
            new_labels = apply_morphological_op(new_labels, *morphopts)

        if overlap > 0:
            sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = \
                     new_labels[overlaps[0][0]:overlaps[0][0]+xr[1]-xr[0],
                                overlaps[1][0]:overlaps[1][0]+yr[1]-yr[0],
                                overlaps[2][0]:overlaps[2][0]+zr[1]-zr[0]]
        else:
            sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = new_labels 
    else:
        sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = 0


def zarr_graphcut3d(source_path, sink_path, **opt):
    '''
    Using zarrs, perform a 3D graph cut 
    
    We take advantage of the zarr file in order to use _saturate_image such that it doesn't 
    take up a bunch of memory. 
    
    Extra option:
    min_threhsold - float, pixel intensity value under which we set to 0 in the image 
    '''

    global temp_zarr_path 
    
    if 'saturate_image_threshold' in opt:
        saturate_image_threshold = opt['saturate_image_threshold']
    else:
        saturate_image_threshold = 0     
    if 'min_threshold' in opt:
        min_threshold = opt['min_threshold']
    else:
        min_threshold = 0
    if 'k' in opt:
        k = opt['k']
    else:
        k = 2
    if 'alpha' in opt:
        alpha = opt['alpha']
    else:
        alpha = 1
    if 'overlap' in opt:
        overlap = opt['overlap']
    else:
        overlap = 0 
    if 'num_workers' in opt:
        num_workers = opt['num_workers']
    else:
        num_workers = 8
    if 'morphopts' in opt:
        morphopts = opt['morphopts']
    else:
        morphopts = None 
    if 'sample_coord_ranges' in opt:
        sample_coord_ranges = opt['sample_coord_ranges']
    else:
        sample_coord_ranges = None 
        
    source_zarr = zarr.open(source_path, mode='r')

    # add compressor (to get rid of blosc issue)
    sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_path), shape=source_zarr.shape, 
                            chunks=source_zarr.chunks, dtype=np.uint8, overwrite=True)#, compressor=Zstd(level=1))
    
    start = time.time()
    print("Starting image preprocessing..")
    if saturate_image_threshold > 0:
        _saturate_image(source_zarr, saturate_image_threshold, min_threshold=min_threshold, num_workers=num_workers) 
        new_source_zarr = zarr.open(temp_zarr_path, mode='r')
    else:
        new_source_zarr = source_zarr
    print("Saturated image processed in %f minutes"%((time.time()-start)/60))
    
    
    
    # Allows users to see if they like what they see before continuing 
    # We sample near the beginning in order to see what the segmentation would look like  for the surface
    if sample_coord_ranges is not None:
        for sample_coord_range in sample_coord_ranges:
            xrs,yrs,zrs = sample_coord_range
            sample_coord_range_test = np.array(get_chunk_coords((xrs[1]-xrs[0],yrs[1]-yrs[0],zrs[1]-zrs[0]), sink_zarr.chunks))
            sample_coord_range_test[:,0,:] += xrs[0] 
            sample_coord_range_test[:,1,:] += yrs[0]
            sample_coord_range_test[:,2,:] += zrs[0] 

            f = partial(_graphcut_kernel_zarr, alpha, k, new_source_zarr, sink_zarr, overlap, morphopts) 
            p = mp.Pool(num_workers)
            list(tqdm(p.imap(f, sample_coord_range_test), total=len(sample_coord_range_test)))
            p.close()
            p.join() 

            # Automatically stack them together OR use Nuggt to display them 
            if morphopts is not None:
                sample_source_path = sink_path[:-5] + '_thresh%d-%d_%s%d_x%d_%d_y%d_%d_z%d_%d'%\
                                        (min_threshold, saturate_image_threshold, morphopts[0], morphopts[2],  
                                         xrs[0],xrs[1],yrs[0],yrs[1],zrs[0],zrs[1]) + '_original.tif'
                sample_sink_path = sink_path[:-5] + '_thresh%d-%d_%s%d_x%d_%d_y%d_%d_z%d_%d'%\
                                    (min_threshold, saturate_image_threshold, morphopts[0], morphopts[2],  
                                     xrs[0],xrs[1],yrs[0],yrs[1],zrs[0],zrs[1]) + '_segmented.tif'
            else:
                sample_source_path = sink_path[:-5] + '_thresh%d-%d_x%d_%d_y%d_%d_z%d_%d'%\
                                    (min_threshold, saturate_image_threshold,   
                                     xrs[0],xrs[1],yrs[0],yrs[1],zrs[0],zrs[1]) + '_original.tif'
                sample_sink_path = sink_path[:-5] + '_thresh%d-%d_x%d_%d_y%d_%d_z%d_%d'%\
                                    (min_threshold, saturate_image_threshold,   
                                     xrs[0],xrs[1],yrs[0],yrs[1],zrs[0],zrs[1]) + '_segmented.tif'
                

            original = source_zarr[xrs[0]:xrs[1],yrs[0]:yrs[1],zrs[0]:zrs[1]]
            segged = sink_zarr[xrs[0]:xrs[1],yrs[0]:yrs[1],zrs[0]:zrs[1]]

            if not USE_NUGGT:
                original = np.expand_dims(original, axis=3)
                segged = np.expand_dims(segged, axis=3)
                io.writeData(sample_sink_path, np.concatenate((original,segged.astype(original.dtype)), axis=3))
            else:
                io.writeData(sample_source_path, original)
                io.writeData(sample_sink_path, segged)
                nuggt_cmd = 'nuggt-display '+sample_source_path+' original red '+sample_sink_path+' segged green --ip-address 10.93.6.101 --port=8900'
                print('Display in Nuggt by doing !{cmd}: \n%s'%(nuggt_cmd))

        # check = input("Continue with these parameters? ([y]/n)")
        # if check == 'n' or check == 'N':
        #     return 
        return 
        
    coord_ranges = get_chunk_coords(sink_zarr.shape, sink_zarr.chunks)  
    f = partial(_graphcut_kernel_zarr, alpha, k, new_source_zarr, sink_zarr, overlap, morphopts) 
    p = mp.Pool(num_workers)
    list(tqdm(p.imap(f, coord_ranges), total=len(coord_ranges)))
    p.close()
    p.join() 



############ Below are tools for saturating an image for easier segmentation ##########    
    
def _saturate_image(image, threshold, min_threshold=0, num_workers=None):
    '''
    Saturates an image for better segmentation. 
    
    Inputs:
    image - 2D or 3D array (or zarr array) 
    thresh - float, fraction of points in image to saturate (the larger the more saturated points)\
    min_threshold - float, pixel intensity value below which to set all pixels to be 0 
    num_workers - if not None, will use parallel_percentile_threshold to avoid memory issues. Otherwise,
                  specifies the number of workers to parallelize saturate_image 
    
    Outputs:
    sat_image - saturated image 
    '''
    global temp_zarr_path 
    
    if num_workers is None:
        if threshold > 1:
            intensity_threshold = threshold 
        else:
            intensity_threshold = _percentile_threshold(image, 1-threshold)
        if min_threshold > intensity_threshold:
            raise RuntimeError('Min_threshold (%f) cannot be greater or equal to the saturation image threshold (%f)'%(min_threshold, intensity_threshold))
        sat_image = _apply_saturation_threshold([image,intensity_threshold,None,min_threshold], [[0,image.shape[0]],[0,image.shape[1]],[0,image.shape[2]]])
        return sat_image 
    else:
        # Assume that the image is a zarr file
        if threshold > 1:
            intensity_threshold = threshold 
        else:
            intensity_threshold = parallel_percentile_threshold(image, 1-threshold, num_workers=num_workers) 
        if min_threshold > intensity_threshold:
            raise RuntimeError('Min_threshold (%f) cannot be greater or equal to the saturation image threshold (%f)'%(min_threshold, intensity_threshold))
        
        # We now apply the saturation 
        chunks = image.chunks # needs to be a zarr 
        coord_ranges = get_chunk_coords(image.shape, chunks) 
                        
        # temporary sink zarr for saturated image outputs
        sink_zarr = zarr.open(temp_zarr_path, mode='w', shape=image.shape, chunks=image.chunks, dtype=image.dtype)  
        p = mp.Pool(num_workers) 
        f = partial(_apply_saturation_threshold, [image, intensity_threshold, sink_zarr, min_threshold])
        p.map(f, coord_ranges)
        p.close()
        p.join() 
        

def _apply_saturation_threshold(args, coord_range):
    '''
    Applies an intensity threshold at which we saturate the pixels in an image.
    '''
    # arg[0] = source_zarr or source image 
    # arg[1] = intensity threshold
    # arg[2] = sink_zarr (if we are using zarrs) - temp 
    # arg[3] = min_threshold, the threshold below which all pixels are set to 0, default 0 
    
    if len(args) == 3:
        source_zarr, intensity_threshold, sink_zarr= args 
        min_threshold = 0 
    elif len(args) == 4:
        source_zarr, intensity_threshold, sink_zarr, min_threshold = args 
    xr,yr,zr = coord_range
    img = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
    saturated_mask = img >= intensity_threshold 
    sat_image = img.copy()
    sat_image[saturated_mask] = intensity_threshold*saturated_mask[saturated_mask]

    if min_threshold > 0:
        sat_image[img < min_threshold ] = 0 
    
    if sink_zarr is not None:
        sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = sat_image 
    else:
        return sat_image    
    
def _percentile_threshold(image, threshold):
    '''
    Calculates the desired intensity threshold based on the image and a pre-specified 
    fraction of pixels to be filtered out, and then thresholds the image. 
    
    Inputs:
    image - nd array 
    thresh - float, fraction of pixels to be thresholded 
    
    Outputs:
    intensity_threshold - the intensity threshold 
    '''
    
    intensity_threshold = np.percentile(image, 100*threshold)
    print("Intensity threshold:",intensity_threshold) 
    return intensity_threshold
    
def parallel_percentile_threshold(source_zarr, threshold, num_workers=1):
    '''
    Parallelizes _percentile_threshold so we dont' run into memory constraints.
    Uses histogram to compute the threshold. 
    
    When num_workers is 1, we do a serial version of memory-saving. 
    '''
    
    # Create a histogram with 65356 bins (one for each pixel value) 
    nbins = 65536
    bin_edges = np.linspace(0,65535,nbins)
    
    # Iterate over zarr based on the chunks 
    chunks = source_zarr.chunks 
    coords_ranges = get_chunk_coords(source_zarr.shape, chunks)
    if num_workers > 1:
        p = mp.Pool(num_workers) 
        f = partial(_get_histogram, [source_zarr, bin_edges])
        hists = p.map(f, coords_ranges)
        total = sum(hists).astype('uint64')
        p.close()
        p.join() 
    else:
        total = np.zeros(nbins, np.uint64)
        for coord_range in coords_ranges:
            total += _get_histogram([source_zarr, bin_edges],coord_range)
    
    # Based on the histogram we can compute the appropriate point at which to threshold 
    # np.save('temp/total.npy',total) 
    cum_totals = np.cumsum(total) 
    cum_total_fractions = cum_totals / sum(total) #cum_totals[-1]
    
    idxs = np.argwhere(cum_total_fractions >= threshold)
    if len(idxs) > 0:
        intensity_threshold = idxs[0]
    else:
        print("Something's wrong with cumulative summing") 
        
    print("Intensity threshold:",intensity_threshold)
    return intensity_threshold
    

def _get_histogram(args, coord_range):
    # arg[0] = source_zarr 
    # arg[1] = bin_edges
    
    source_zarr, bin_edges = args 
    xr,yr,zr = coord_range
    img = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
    subtotal, _ = np.histogram(img, bins=bin_edges)
    #print("Got histogram of slices %d"%(zr[0]))
    return subtotal.astype(np.uint64)


def equalize_hist(img, nbins=256, mask=None, num_workers=None):
    """
    Serial and multiprocesing version of histogram equalization 

    """
    if num_workers is None or num_workers <= 1:
        if type(img) != np.ndarray:
            try:
                img_numpy = img[:] 
                return exposure.equalize_hist(img_numpy, nbins=nbins, mask=mask)
            except:
                print("img needs to be a numpy ndarray or zarr")
        else:
            return exposure.equalize_hist(img, nbins=nbins, mask=mask)
    else:
        # multiprocessing 
        # this requires img to be a zarr file 
        bin_edges = np.linspace(0,65535,nbins)
        return 


'''
def apply_mclahe(img, kernel_size=None, n_bins=128, clip_limit=0.01, adaptive_hist_range=False, use_gpu=False):
    """
    Contrast limited adaptive histogram equalization implemented in tensorflow
    :param x: numpy array to which clahe is applied
    :param kernel_size: tuple of kernel sizes, 1/8 of dimension lengths of x if None
    :param n_bins: number of bins to be used in the histogram
    :param clip_limit: relative intensity limit to be ignored in the histogram equalization
    :param adaptive_hist_range: flag, if true individual range for histogram computation of each block is used
    :param use_gpu: Flag, if true gpu is used for computations if available
    :return: numpy array to which clahe was applied, scaled on interval [0, 1]
    """

    return mclahe(img, kernel_size, n_bins, clip_limit, adaptive_hist_range, use_gpu)
'''
    
    
def threshold_image(source_zarr_path, sink_zarr_path, percentile_threshold, 
                    threshold_type='set', morphopts=None, num_workers=1, sample_coord_ranges=None):
    '''
    Function for applying intensity threshold to large zarr/images.
    
    Inputs:
    threshold_type - str, can be 'set' (user sets the value in percentile threshold)
                          or can be 'otsu' (Otsu adaptive thresholding) 
    morphopts - if not None:  
            tuple(str, str, int), option for post-processing the graph cut segmentation.
             str1: option for morphological operation
             str2: structuring element type  
             Int: size fo the structuring element
             See apply_morphological_op for more descriptions.
    '''
    source_zarr = zarr.open(source_zarr_path, mode='r')
    sink_zarr = zarr.open(sink_zarr_path, mode='w', shape=source_zarr.shape, 
                          chunks=source_zarr.chunks, dtype=source_zarr.dtype) 

    if sample_coord_ranges is not None:
        for sample_coord_range in sample_coord_ranges:
            if threshold_type == 'set':         
                if percentile_threshold < 1:
                    intensity_threshold = parallel_percentile_threshold(source_zarr, percentile_threshold, num_workers=num_workers)
                else:
                    intensity_threshold = percentile_threshold

            xrs,yrs,zrs = sample_coord_range
            sample_coord_range_test = np.array(get_chunk_coords((xrs[1]-xrs[0],yrs[1]-yrs[0],zrs[1]-zrs[0]), sink_zarr.chunks))
            sample_coord_range_test[:,0,:] += xrs[0] 
            sample_coord_range_test[:,1,:] += yrs[0]
            sample_coord_range_test[:,2,:] += zrs[0] 

            if threshold_type == 'set':
                args = (source_zarr, intensity_threshold, sink_zarr, morphopts)
                f = partial(_apply_intensity_threshold, args) 
            elif threshold_type == 'otsu':
                f = partial(_apply_otsu_threshold, source_zarr, sink_zarr)


            p = mp.Pool(num_workers)
            list(tqdm(p.imap(f, sample_coord_range_test), total=len(sample_coord_range_test)))
            p.close()
            p.join() 

            if morphopts is not None:
                sample_sink_path = sink_zarr_path[:-5] + '_thresh%d_%s%d_x%d_%d_y%d_%d_z%d_%d'%\
                                    (intensity_threshold, morphopts[0], morphopts[2],  
                                     xrs[0],xrs[1],yrs[0],yrs[1],zrs[0],zrs[1]) + '_segmented.tif'
            else:
                sample_sink_path = sink_zarr_path[:-5] + '_thresh%d_x%d_%d_y%d_%d_z%d_%d'%\
                                    (intensity_threshold,   
                                     xrs[0],xrs[1],yrs[0],yrs[1],zrs[0],zrs[1]) + '_segmented.tif'
            segged = sink_zarr[xrs[0]:xrs[1],yrs[0]:yrs[1],zrs[0]:zrs[1]]
            io.writeData(sample_sink_path, segged)
        return 
    else:
        if threshold_type == 'set':         
            if percentile_threshold < 1:
                intensity_threshold = parallel_percentile_threshold(source_zarr, percentile_threshold, num_workers=num_workers)
            else:
                intensity_threshold = percentile_threshold 
                
        coord_ranges = get_chunk_coords(sink_zarr.shape, sink_zarr.chunks)
        p = mp.Pool(num_workers) 
        if threshold_type == 'set':
            f = partial(_apply_intensity_threshold, [source_zarr, intensity_threshold, sink_zarr, morphopts])
        elif threshold_type == 'otsu':
            f = partial(_apply_otsu_threshold, source_zarr, sink_zarr)
        
        list(tqdm(p.imap(f, coord_ranges, chunksize=10), total=len(coord_ranges)))
        p.close()
        p.join() 
    
    
def _apply_intensity_threshold(args, coord_range):
    '''
    Applies an intensity threshold at which we saturate the pixels in an image.
    '''
    # arg[0] = source_zarr or source image 
    # arg[1] = intensity threshold
    # arg[2] = sink_zarr (if we are using zarrs) 
    # arg[3] = morphopts (if None, don't apply any)

    source_zarr, intensity_threshold, sink_zarr, morphopts = args 
    xr,yr,zr = coord_range
    img = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
    if np.any(img):
        mask = img >= intensity_threshold 

        if morphopts is not None:
            mask = apply_morphological_op(mask, *morphopts)

        if sink_zarr is not None:
            sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = mask  
        else:
            return mask  
    else:
        sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = 0
        
def _apply_otsu_threshold(source_zarr, sink_zarr, coord_range):
    '''
    For calculating and applying OTSU threshold
    '''
    xr,yr,zr = coord_range 
    img = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
    if np.any(img > 0):
        thresh = threshold_otsu(img)
    else:
        thresh = 1.0 
    mask = img >= thresh 
    if sink_zarr is not None:
        sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = mask         
    else:
        return mask  

## TODO: function to apply morphological operations 
        
        
################### The following are for finding surfaces ########################



def mask2surf(mask, opening_size=None, num_slices_to_search=None, num_slices_to_add=None):
    '''
    Gets the top surface of the tissue based on segmented mask.
    TODO: make functionality to find surface at the bottom too 
    
    Inputs:
    mask (numpy ndarray) - the image mask (of dtype uint8 with only 1's and 0's)
    openine_size (int, int) - the size of the morphological opening element with which we get rid of small elements (default: None)
    num_slices_to_search (int) - number of z slices to look for the surface in. (default: None) 
    num_slices_to_add (int) - number of z slices to MIP for the surface (default: None) 
    
    Outputs:
    surface (numpy ndarray) - a mask containing only pixels on the surface 
    '''
    mask = (mask > 0).astype('int') 
    if num_slices_to_search is None:
        num_slices_to_search = mask.shape[2]-num_slices_to_add
    elif num_slices_to_add is not None:
        if num_slices_to_search + num_slices_to_add > mask.shape[2]:
            num_slices_to_search = mask.shape[2] - num_slices_to_add 
            
    surface = np.zeros(mask[:,:,:num_slices_to_search+num_slices_to_add].shape,dtype='bool')
    surface[:,:,0] = mask[:,:,0].copy()
    # First create the thin one layer surface 
    for i in range(1, num_slices_to_search):
        running_total = (np.sum(mask[:,:,:i],axis=2) > 0).astype('int')
        surface[:,:,i] = (mask[:,:,i] - running_total) > 0 
    
    dilated_surface = surface.copy()
    if num_slices_to_add is not None or num_slices_to_add > 0:
        surf_inds = np.argwhere(surface) # where the surface indices are 
        for i in range(0, num_slices_to_add):
            dilated_surface[surf_inds[:,0], surf_inds[:,1], surf_inds[:,2]+i+1] = 1
            
    # Apply morphological opening to get rid of small pixels 
    # if opening_size is not None:
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_size)
    #     for i in range(dilated_surface.shape[2]):   
    #         dilated_surface[:,:,i] = cv2.morphologyEx(dilated_surface[:,:,i].astype('uint8'), cv2.MORPH_OPEN, kernel)
        
    # Functionality to have areas above the surface be counted as the surface for better endpoint detection
    for i in np.arange(surface.shape[2]):
        slice_num = surface.shape[2]-i-1
        inds_temp = np.argwhere(surface[:,:,slice_num])
        dilated_surface[inds_temp[:,0],inds_temp[:,1],:slice_num] = 1

    return dilated_surface.astype('uint8')
    

    


def mask2surf_zarr(mask_zarr_path, surf_zarr_path, slices_to_search, num_slices_to_add, 
                    num_slices_above=False,orientation='ztop',use_full_image=False,
                    thin_surf_zarr_path=None,num_workers=None):
    '''
    Same as mask2surf but parallelized with zarr functionality.
    slices_to_search: [z_start, z_end], which z slices to search

    num_slices_above: int or bool, number of slices to add above the surface to include in the surface. 
                        If False, then does not execute. If True, will include all voxels above the surface.
                        If int, then includes num_slices_above slices into the surface.
    orientation: str, 'top' or 'bottom', if 'top', then detect the surface from z=0, whereas if 'bottom', detect from z=bottom
    '''
    global temp_zarr_path 
    
    if not use_full_image:
        mask_zarr = zarr.open(mask_zarr_path, mode='r')
    else:
        # This makes it faster but may not be memory efficient 
        mask_zarr = zarr.open(mask_zarr_path, mode='r')[:]

    z1,z2 = slices_to_search 
    num_slices_to_search = z2-z1 

    if orientation[0] == 'z':
        dim = 2 
    elif orientation[0] == 'y':
        dim = 1
    elif orientation[0] == 'x':
        dim = 0

    if num_slices_to_search + num_slices_to_add > mask_zarr.shape[dim]:
        num_slices_to_search = mask_zarr.shape[dim] - num_slices_to_add 
        z2 = z1 + num_slices_to_search
    

    if not use_full_image:
        surf_zarr = zarr.zeros(shape = mask_zarr.shape, #(*mask_zarr.shape[:2],mask_zarr.shape[2]),#z2-z1+num_slices_to_add),
                                chunks = mask_zarr.chunks,
                                dtype=np.uint8,
                                store=zarr.DirectoryStore(surf_zarr_path), overwrite=True)
        # The following will be stored in the temp folder - store the thin surface layer 
        if thin_surf_zarr_path is None:
            thin_surf_zarr_path = temp_zarr_path 
        thin_surf_zarr = zarr.zeros(shape = surf_zarr.shape, 
                               chunks = surf_zarr.chunks,
                               dtype=np.uint8,
                               store=zarr.DirectoryStore(thin_surf_zarr_path), overwrite=True) 
    else:
        surf_zarr = np.zeros(mask_zarr.shape, dtype='uint8')
        thin_surf_zarr = np.zeros(surf_zarr.shape, dtype='uint8')

    
    ## Serial version 
    if num_workers is None:
        #running_total = np.zeros(surf_zarr.shape[:2],dtype='int')
        if orientation[0] == 'z':
            running_total = np.zeros(surf_zarr.shape[:2], dtype='int')
        elif orientation[0] == 'y':
            running_total = np.zeros((surf_zarr.shape[0],surf_zarr.shape[2]), dtype='int')
        elif orientation[0] == 'x':
            running_total = np.zeros((surf_zarr.shape[1],surf_zarr.shape[2]), dtype='int')

        if orientation =='ztop' or orientation == 'xtop' or orientation == 'ytop':
            slices = np.arange(z1,z2)
        elif orientation=='zbottom' or orientation == 'xbottom' or orientation == 'ybottom':
            slices = np.flip(np.arange(z1,z2))

        for i in tqdm(slices): # now that we're ont starting from 0
            if orientation[0] == 'z':
                thin_surf_zarr[:,:,i] = (mask_zarr[:,:,i] - running_total) > 0 
                running_total += mask_zarr[:,:,i]
                surf_inds = np.argwhere(thin_surf_zarr[:,:,i])
                # Assign the buffer region 
                if orientation[1:] == 'top':
                    temp_surf_npy = surf_zarr[:,:,i:i+num_slices_to_add]
                    temp_surf_npy[surf_inds[:,0],surf_inds[:,1],:] = 1
                    surf_zarr[:,:,i:i+num_slices_to_add] = temp_surf_npy
                elif orientation[1:] == 'bottom':
                    temp_surf_npy = surf_zarr[:,:,i-num_slices_to_add:i]
                    temp_surf_npy[surf_inds[:,0],surf_inds[:,1],:] = 1
                    surf_zarr[:,:,i-num_slices_to_add:i] = temp_surf_npy

            elif orientation[0] == 'y':
                thin_surf_zarr[:,i,:] = (mask_zarr[:,i,:] - running_total) > 0
                running_total += mask_zarr[:,i,:]
                surf_inds = np.argwhere(thin_surf_zarr[:,i,:])
                # Assign the buffer region 
                if orientation[1:] == 'top':
                    temp_surf_npy = surf_zarr[:,i:i+num_slices_to_add,:]
                    temp_surf_npy[surf_inds[:,0],:,surf_inds[:,1]] = 1
                    surf_zarr[:,i:i+num_slices_to_add,:] = temp_surf_npy
                elif orientation[1:] == 'bottom':
                    temp_surf_npy = surf_zarr[:,i-num_slices_to_add:i,:]
                    temp_surf_npy[surf_inds[:,0],:,surf_inds[:,1]] = 1
                    surf_zarr[:,i-num_slices_to_add:i,:] = temp_surf_npy

            elif orientation[0] == 'x':
                thin_surf_zarr[i,:,:] = (mask_zarr[i,:,:] - running_total) > 0 
                running_total += mask_zarr[i,:,:]
                surf_inds = np.argwhere(thin_surf_zarr[i,:,:])
                # Assign the buffer region 
                if orientation[1:] == 'top':
                    temp_surf_npy = surf_zarr[i:i+num_slices_to_add,:,:]
                    temp_surf_npy[:,surf_inds[:,0],surf_inds[:,1]] = 1
                    surf_zarr[i:i+num_slices_to_add,:,:] = temp_surf_npy
                elif orientation[1:] == 'bottom':
                    temp_surf_npy = surf_zarr[i-num_slices_to_add:i,:,:]
                    temp_surf_npy[:,surf_inds[:,0],surf_inds[:,1]] = 1
                    surf_zarr[i-num_slices_to_add:i,:,:] = temp_surf_npy
             
        


        # Now we make sure to also assign all areas above the surface to be "part of the surface" also.
        # This ensures that even with imperfect surface detection we can capture the endpoints 
        # Start the for loop in reverse of the thin_surf_zarr 
        if num_slices_above:
            print('Adding slices above...')
            running_total = np.zeros(surf_zarr.shape[:2],dtype='int')
            if num_slices_above is True:
                for i in tqdm(np.arange(thin_surf_zarr.shape[2]-1)):
                    if orientation == 'ztop':
                        slice_num = surf_zarr.shape[2]-i-2
                        running_total += thin_surf_zarr[:,:,slice_num+1]
                    elif orientation == 'zbottom':
                        slice_num = i+1
                        running_total += thin_surf_zarr[:,:,slice_num-1]
                    surf_zarr[:,:,slice_num] = (running_total + surf_zarr[:,:,slice_num]) > 0 
            elif num_slices_above: # if an actual number was assigned 
                for i in tqdm(np.arange(surf_zarr.shape[2]-1)):
                    if orientation == 'ztop':
                        slice_num = surf_zarr.shape[2]-i-2
                        ainds = np.argwhere(thin_surf_zarr[:,:,slice_num+1]) # key difference is we use the unchanged thin surface
                        begin_slice_num = slice_num-num_slices_above+1 
                        begin_slice_num = begin_slice_num * (begin_slice_num > 0) # if it's negative, then the beginning slice is 0
                        temp_surf_npy = surf_zarr[:,:,begin_slice_num:slice_num+1]
                        temp_surf_npy[ainds[:,0],ainds[:,1]] = 1
                        surf_zarr[:,:,begin_slice_num:slice_num+1] = temp_surf_npy
                    elif orientation == 'zbottom':
                        slice_num = i+1
                        ainds = np.argwhere(thin_surf_zarr[:,:,slice_num-1])
                        begin_slice_num = slice_num + num_slices_above 
                        begin_slice_num *= (begin_slice_num < surf_zarr.shape[2])
                        temp_surf_npy = surf_zarr[:,:,slice_num:begin_slice_num]
                        temp_surf_npy[ainds[:,0],ainds[:,1]] = 1
                        surf_zarr[:,:,slice_num:begin_slice_num] = temp_surf_npy


        if use_full_image:
            mask_zarr = zarr.open(mask_zarr_path, mode='r')
            surf_zarr_ = zarr.zeros(shape = mask_zarr.shape, 
                                chunks = mask_zarr.chunks,
                                dtype=np.uint8,
                                store=zarr.DirectoryStore(surf_zarr_path), overwrite=True)
            surf_zarr_[:] = surf_zarr 
            if thin_surf_zarr_path is not None:
                thin_surf_ = zarr.zeros(shape = surf_zarr_.shape, 
                               chunks = surf_zarr_.chunks,
                               dtype=np.uint8,
                               store=zarr.DirectoryStore(thin_surf_zarr_path), overwrite=True) 
                thin_surf_[:] = thin_surf_zarr

    else: # TODO: parallelize  
        return 

def polygon_mask_surface(contour_pts_path, surf_zarr_paths, json_name=None, downsample_factor=1,
                         surf_zarr_save_paths=None, polygon_save_path=None):
    '''
    Read in points representing the contour of a surface (in 2D)
    - contour_pts_path can be json (annotation from Neuroglancer), 
    Mask a detected 3D surface using that contour 
    '''

    if contour_pts_path[-4:] == 'json':
        new_concave_hull = np.round(read_annotations_json(contour_pts_path, json_name, sink_path=None)).astype('int')
    elif contour_pts_path[-3:] == 'npy':
        new_concave_hull = np.round(np.load(contour_pts_path)).astype('int')


    if surf_zarr_paths[0][-4:] == 'zarr':
        img_size = zarr.open(surf_zarr_paths[0],mode='r').shape 
    else:
        img_size = io.readData(surf_zarr_paths[0]).shape 

    if contour_pts_path[-3:] == 'tif' or contour_pts_path[-4:] == 'tiff':
        mask_ = io.readData(contour_pts_path)
        # resample to be same size
        mask = ndi.zoom(mask_,float(1/downsample_factor))
        # print(mask.shape)
    else:
        img = Image.new('L', img_size[:2], 0)
        ImageDraw.Draw(img).polygon([tuple(np.round(temp_/downsample_factor).astype('int')) for temp_ in new_concave_hull[:,:2]], outline=1, fill=1)
        mask = np.array(img).T
        if polygon_save_path is not None:
                io.writeData(polygon_save_path,mask)


    surf_imgs = []
    for idx,zarr_path in enumerate(surf_zarr_paths):
        if zarr_path[-4:] == 'zarr':
            surf_img = zarr.open(zarr_path,mode='r')[:] 
        elif zarr_path[-3:] == 'tif' or zarr_path[-4:] == 'tiff':
            surf_img = io.readData(zarr_path)

        surf_mask_3d = np.transpose(np.array([mask]*surf_img.shape[2]),[1,2,0])
        new_surf = surf_mask_3d*surf_img
        surf_imgs.append(new_surf)

        if surf_zarr_save_paths is not None:
            if surf_zarr_save_paths[idx][-4:] == 'zarr':
                z_ = zarr.create(store=zarr.DirectoryStore(surf_zarr_save_paths[idx]), shape=surf_img.shape, 
                                chunks=zarr.open(zarr_path).chunks, dtype=np.uint8, overwrite=True)
                z_[:] = new_surf
            elif surf_zarr_save_paths[idx][-3:] == 'tif' or surf_zarr_save_paths[idx][-4:] == 'tiff':
                io.writeData(surf_zarr_save_paths[idx],new_surf)
    
    return surf_imgs 

def mask_point_cloud(eps_path, mask_path, img_shape, eps_save_path=None, resample_factor=1):
    '''
    Filter out points in point cloud that do not lie in an image mask.
    '''
    eps = np.load(eps_path)
    mask = io.readData(mask_path)
    eps_upsampled = np.round(eps*resample_factor).astype('int')
    if len(img_shape) == 2:
        eps_img = pts_to_image(eps_upsampled[:,:2], img_shape)
    else:
        eps_img = pts_to_image(eps_upsampled, img_shape)

    eps_img_masked = eps_img*mask
    eps_filtered = np.argwhere(eps_img_masked)

    final_eps = np.zeros((0,3),dtype='int')
    for ep in eps_filtered:
        if len(img_shape)==2:
            final_eps = np.concatenate((final_eps,eps[(eps_upsampled[:,0]==ep[0])*(eps_upsampled[:,1]==ep[1])]),axis=0)
        else:
            final_eps = np.concatenate((final_eps,eps[(eps_upsampled[:,0]==ep[0])*\
                                                    (eps_upsampled[:,1]==ep[1])*\
                                                    (eps_upsampled[:,2]==ep[2])]),axis=0)
    
    if eps_save_path is not None:
        np.save(eps_save_path, final_eps)
    return final_eps

def mip(surface, img):
    '''
    Generate a maximum intensity projection of the surface 
    '''
    img = img[:,:,:surface.shape[2]] 
    surf = surface*img 
    mip = np.max(surf, axis=2)
    #args = np.argmax(surf, axis=2) 
    print(mip.shape) 
    ## Get segmented MIP
    #mip_segment = np.sum(surf, axis=2) > 0 
    return mip #, mip_segment.astype('uint8'), args

    
################# For combining mamsks 
        
def union_masks(masks):
    '''
    Perform union of list of images which are masks 
    '''
    masks = np.array(masks)
    masks_added = np.sum(masks,axis=0) > 0
    return masks_added.astype('uint8')

def _union_masks_pkernel(mask_zarrs, sink_zarr, coord_range):
    '''
    parallel kernel for processing large datasets in parallel
    '''
    xr,yr,zr = coord_range
    masks = [mask_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] for mask_zarr in mask_zarrs]
    masks_added = np.sum(np.array(masks),axis=0) > 0
    sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = masks_added.astype('uint8')

def union_masks_zarr(mask_zarr_paths, sink_zarr_path, num_workers=8, chunks=None):
    '''
    Perform union masks but parallel for zarrs.
    '''
    mask_zarrs = [zarr.open(mask_zarr_path, mode='r') for mask_zarr_path in mask_zarr_paths]

    if chunks is None:
        chunks = (np.min([mask_zarr.chunk[0] for mask_zarr in mask_zarrs]),
                  np.min([mask_zarr.chunk[1] for mask_zarr in mask_zarrs]),
                  np.min([mask_zarr.chunk[2] for mask_zarr in mask_zarrs]))
    coord_ranges = get_chunk_coords(mask_zarrs[0].shape, chunks)

    sink_zarr = zarr.open(sink_zarr_path,mode='w',shape=mask_zarrs[0].shape,chunks=chunks,dtype='uint8')

    p = mp.Pool(num_workers) 
    f = partial(_union_masks_pkernel, mask_zarrs, sink_zarr)
    list(tqdm(p.imap(f, coord_ranges), total=len(coord_ranges)))
    p.close()
    p.join()



############## max intensity projections

def mip_zarr(img_zarr_path, surf_zarr_path=None, num_slices=None, first_slice=None, num_workers=None):
    '''
    Gets max intensity projection for large volumes.
    
    If surf_zarr is none, then we just take a certain number of slices 
    If num_slices is None, then the function expects a surface to be specified
    if first_slice is None and surf_zarr_path is not None, then we start the surface 
        from the first slice. Otherwise, if both are not None, then we start finding
        the surface from first_slice 
    '''
    img = zarr.open(img_zarr_path, mode='r') 
    if first_slice is None:
        first_slice = 0
    if surf_zarr_path is not None:
        surf = zarr.open(surf_zarr_path, mode='r')
        chunk_coords = get_chunk_coords(surf.shape, surf.chunks) 
        if num_workers is None:
            mip = np.zeros(surf.shape[:2], dtype='uint16')
            for chunk in chunk_coords:
                xr,yr,zr = chunk
                if xr[1] > surf.shape[0]:
                    xr[1] = surf.shape[0]
                if yr[1] > surf.shape[1]:
                    yr[1] = surf.shape[1]
                if zr[1] > surf.shape[2]:
                    zr[1] = surf.shape[2] 
                curr_mip = np.max(surf[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] *\
                                  img[xr[0]:xr[1],yr[0]:yr[1],zr[0]+first_slice:zr[1]+first_slice],axis=2)
                mip = np.maximum(mip,curr_mip)
                print('Done with slice %d'%zr[0])
        else:
            return 
    elif num_slices is not None:
        chunk_coords = get_chunk_coords((*img.shape[:2],num_slices),img.chunks) 
        if num_workers is None: 
            mip = np.zeros((*img.shape[:2],num_slices),dtype='uint16')
            for chunk in chunk_coords:
                xr,yr,zr = chunk 
                curr_mip = np.max(img[xr[0]:xr[1],yr[0]:yr[1],zr[0]+first_slice:zr[1]+first_slice],axis=2)
                mip = np.maximum(mip,curr_mip) 
        else:
            return 
    return mip.astype('uint16')


######################## Miscellaneous ##########################

# Gaussian smoothing 
def gaussian3d(img, sigma):
    '''
    skimage implementation of gaussian smoothing

    Inputs:
    img - numpy ndarray
    sigma - std deviation
    '''
    return gaussian(img, sigma=sigma)

def _gaussian3d_kernel(source_zarr, sink_zarr, sigma, coords):
    xr,yr,zr = coords 
    img = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
    sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = (gaussian3d(img, sigma)*65535).astype('uint16')


def gaussian3d_zarr(source_zarr_path, sink_zarr_path, num_workers, sigma=1.0):
    '''
    Parallel Gaussian smoothing of zarr image
    '''

    source_zarr = zarr.open(source_zarr_path, mode='r')
    sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_zarr_path), shape=source_zarr.shape, 
                            chunks=source_zarr.chunks, dtype=np.uint16, overwrite=True)
    coords = get_chunk_coords(sink_zarr.shape, sink_zarr.chunks) 
    f = partial(_gaussian3d_kernel, source_zarr, sink_zarr, sigma) 
    p = mp.Pool(num_workers)
    list(tqdm(p.imap(f, coords), total=len(coords)))
    p.close()
    p.join() 


##########################

# use 1D sobel filter to detect the surface 

###############################

# Potentially a better way to do 3D surface detection 

def ellipsoid(xr, yr, zr):
    '''
    Makes an ellipsoid structuring element for morphological operations

    Inputs:
    xr,yr,zr - radius in x,y,z direction

    Outputs:
    structuring element of size (2*xr+1,2*yr+1,2*zr+1)
    '''
    dim = np.max([xr,yr,zr])
    X,Y,Z = np.ogrid[0:2*dim+1, 0:2*dim+1, 0:2*dim+1]
    ellipsoi = (((X-dim)**2/xr**2 + (Y-dim)**2/yr**2 + (Z-dim)**2/zr**2) <= 1.0).astype('uint8')
    return ellipsoi

def bwperim3d(img, radius):
    '''
    Compute the binary perimeter of a binary image.
    
    Inputs:
    img - numpy ndarray, a binary image (segmentation mask)
    radius - structuring element (ball) radius. if tuple, then ellipsoid 
    '''
    if type(radius) == int or type(radius) == float:
        xr = radius; yr = radius; zr = radius 
    else:
        xr,yr,zr = radius 
    selem = ellipsoid(xr,yr,zr)
    eroded_img = morphology.binary_erosion(img, selem=selem)
    perim = img - eroded_img 
    return perim 

def bwperim3d_zarr(source_path, sink_path, num_workers=8):
    '''
    Parallel version for bwperim3d
    '''
    return 
