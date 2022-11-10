"""
LightsheetCorrection
====================

Module to remove lightsheet artifacts in images.
Adapted from Christoph Kirst, 2020
"""


import numpy as np
import scipy.ndimage as ndi 
import zarr
from functools import partial
import multiprocessing as mp
from tqdm import tqdm 
from .utils import *

###############################################################################
### Lightsheet correction
###############################################################################

def correct_lightsheet(source, percentile = 0.25, max_bin=2**12, mask=None,
                       lightsheet = dict(selem = (150,1,1)), 
                       background = dict(selem = (200,200,1), spacing = (25,25,1), interpolate = 1, dtype = float, step = (2,2,1)),
                       lightsheet_vs_background = 2, return_lightsheet = False, return_background = False):
  """Removes lightsheet artifacts.
  
  Arguments
  ---------
  source : array
    The source to correct.
  percentile : float in [0,1]
    Ther percentile to base the lightsheet correction on.
  max_bin : int 
    The maximal bin to use. Max_bin needs to be >= the maximal value in the 
    source.
  mask : array or None
    Optional mask.
  lightsheet : dict
    Parameter to pass to the percentile routine for the lightsheet artifact
    estimate. See :func:`ImageProcessing.Filter.Rank.percentile`.
  background : dict
    Parameter to pass to the percentile rouitne for the background estimation.
  lightsheet_vs_background : float
    The background is multiplied by this weight before comparing to the
    lightsheet artifact estimate.
  return_lightsheet : bool
    If True, return the lightsheeet artifact estimate.
  return_background : bool
    If True, return the background estimate.
  
  Returns
  -------
  corrected : array
    Lightsheet artifact corrected image.
  lightsheet : array
    The lightsheet artifact estimate.
  background : array
    The background estimate.
  
  Note
  ----
  The routine implements a fast but efftice way to remove lightsheet artifacts.
  Effectively the percentile in an eleoganted structural element along the 
  lightsheet direction centered around each pixel is calculated and then
  compared to the percentile in a symmetrical box like structural element 
  at the same pixel. The former is an estimate of the lightsheet artifact 
  the latter of the backgrond. The background is multiplied by the factor 
  lightsheet_vs_background and then the minimum of both results is subtracted
  from the source.
  Adding an overall background estimate helps to not accidentally remove
  vessesl like structures along the light-sheet direction.
  """
  # if verbose:
  #   timer = tmr.Timer();
  
  #lightsheet artifact estimate
  # l =  cm_percentile(source, percentile=percentile, max_bin=max_bin, mask=mask, **lightsheet)

  l = local_percentile(source, percentile=percentile, mask=mask, **lightsheet)
  
  #background estimate                         
  b = local_percentile(source, percentile=percentile, mask=mask, **background);

    
  #combined estimate                                                                                    
  lb = np.minimum(l, lightsheet_vs_background * b);
  
  #corrected image                                           
  c = source - np.minimum(source, lb);

  
  result = (c,)
  if return_lightsheet:
    result += (l,)                    
  if return_background:
    result += (b,) 
  if len(result) == 1:
    result = result[0]     
  return result


def zarr_correct_kernel(func, source_zarr, sink_zarr, chunk_coords, **kwargs):
  xr,yr,zr = chunk_coords 
  img_source = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
  sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = func(img_source, **kwargs)


def parallel_zarr_correct(func, source_zarr_path, sink_zarr_path, num_workers=8, **kwargs):
  '''
  Parallel version of destriping.
  Currently no functionality to return the background or lightsheet artifacts as separate zarr.
  '''

  source_zarr = zarr.open(source_zarr_path, mode='r')
  sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_zarr_path), shape=source_zarr.shape, 
                            chunks=source_zarr.chunks, dtype='uint16', overwrite=True)

  coord_ranges = get_chunk_coords(source_zarr.shape, source_zarr.chunks) 
  f = partial(zarr_correct_kernel, func, source_zarr, sink_zarr, **kwargs) 
  p = mp.Pool(num_workers)
  list(tqdm(p.imap(f, coord_ranges), total=len(coord_ranges)))
  p.close()
  p.join() 


def correct_background(source, percentile = 0.25, mask=None, background = dict(selem = (200,200,1)), factor=1, return_background=False):

  """Removes lightsheet artifacts.
  
  Arguments
  ---------
  source : array
    The source to correct.
  percentile : float in [0,1]
    Ther percentile to base the lightsheet correction on.
  mask : array or None
    Optional mask.
  background : dict
    Parameter to pass to the percentile rouitne for the background estimation.
  factor : float
    Multiply this by the computed percentile intensity value 
  
  Returns
  -------
  corrected : array
    Background corrected image.
  """
  if np.prod(source.shape) != 0:
    #background estimate                         
    b = local_percentile(source, percentile=percentile, mask=mask, **background);

    #corrected image                                           
    c = source - np.minimum(source, b*factor)  
    c = c.astype('uint16')
    if not return_background:  
      return c
    else:
      return c,b
  else:
    return

################# Parallelize this ###################
def _correct_background_kernel(shm, zcoords, **kwargs):
  with shm.txn() as a:
    img = a[:,:,zcoords[0]:zcoords[1]]
  img_corr = correct_background(img, **kwargs)
  return img_corr 

def parallel_correct_background(source_zarr_path, sink_zarr_path, load_z_slices=None, num_workers=24,**kwargs):
  '''
  Perform background correct on zarr, but we want to load in perform to a full lateral section for accurate results

  load_z_slices - int, if not None, the number of slices to load in; if None, then load in the z_chunk size
  '''

  source_zarr = zarr.open(source_zarr_path, mode='r')
  size = source_zarr.shape 
  sink_zarr = zarr.create(store=zarr.DirectoryStore(sink_zarr_path), shape=size, 
                            chunks=source_zarr.chunks, dtype='uint16', overwrite=True)

  if load_z_slices is None:
    load_z_slices = source_zarr.chunks[2]

  DataFileRange = [{'x':[0,size[0]],'y':[0,size[1]],'z':[i,i+load_z_slices]} for i in np.arange(0,size[2],load_z_slices)]
  DataFileRange[-1]['z'][1] = size[2] # correct for this 
  
  for i in range(len(DataFileRange)):
    # First load in zarr in parallel 
    z,z_final = DataFileRange[i]['z']
    print("Working on z=%d to %d"%(z,z_final))
    shm= SharedMemory((*size[:2],z_final-z), sink_zarr.dtype)
    parallel_readzarr(source_zarr_path, num_workers, shm=shm, **DataFileRange[i])

    # Do background correction 
    zcoords = [[j,j+1] for j in range(z_final-z)]
    p = mp.Pool(num_workers)
    f = partial(_correct_background_kernel, shm, **kwargs)
    processed_chunks = list(tqdm(p.imap(f, zcoords), total=len(zcoords)))
    p.close()
    p.join()

    # Finally, let's assign to zarr 
    large_chunk = fast_numpy_concatenate(processed_chunks, axis=2, num_workers=num_workers)

    # Serial version
    sink_zarr[:,:,z:z_final] = large_chunk 

    ## Parallel version 
    # shm2 = SharedMemory((*size[:2],z_final-z), sink_zarr.dtype)
    # small_chunk_coords = get_chunk_coords((*size[:2],z_final-z), sink_zarr.chunks)
    # global_coords = np.asarray(small_chunk_coords)
    # global_coords[:,2,:] += z 
    # global_coords = global_coords.tolist()
    # arrays = [(shm2, (x,y,w)) for (x,y,w) in small_chunk_coords]
    # arrays_and_coords = list(map(lambda q,r:(q,r), arrays, global_coords))

    # p = mp.Pool(num_workers)
    # f = partial(_numpy_to_zarr, sink_zarr)
    # list(tqdm(p.imap(f, arrays_and_coords), total=len(arrays_and_coords)))
    # p.close()
    # p.join()

  
def _temp_readzarr(filename, coords, shm=None):
    '''
    Inputs:
    shm - if not None, then is SharedMemory array into which we assign the arrays
    '''
    local_coords, global_coords = coords
    xr,yr,zr = local_coords
    xg, yg, zg = global_coords
    zarrr = zarr.open(filename,mode='r')
    img = zarrr[xg[0]:xg[1],yg[0]:yg[1],zg[0]:zg[1]]
    if shm is not None:
        with shm.txn() as a:
            a[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = img 
        return 
    else:
        return img 
    

def parallel_readzarr(fname, num_workers, shm=None, **DataFileRange):
    '''
    Speeding up io.readData using parallel processing.

    fname - str, name of zarr 
    num_workers - int, number of workers for parallel processing
    shm - SharedMemory, a SharedMemory array to store the array in. If None, this function returns a numpy array 
    '''
    source_zarr = zarr.open(fname,mode='r')

    z0,z1 = DataFileRange['z']
    x0,x1 = DataFileRange['x']
    y0,y1 = DataFileRange['y']

    size = (x1-x0,y1-y0,z1-z0)
    local_coords = get_chunk_coords(size, source_zarr.chunks)
    coords = np.asarray(local_coords)
    coords[:,0,:] += x0; coords[:,1,:] += y0; coords[:,2,:] += z0
    coords = coords.tolist()

    total_coords = tuple(zip(local_coords, coords))
    p = mp.Pool(num_workers)
    f = partial(_temp_readzarr, fname, shm=shm)
    processed_chunks = list(tqdm(p.imap(f, total_coords), total=len(total_coords)))
    p.close()
    p.join()
    
    if shm is None:
        return processed_chunks 
    else:
        return


  ###############################################################################
### Local image processing for Local Statistics
###############################################################################
    
def apply_local_function(source, function, selem = (50,50), spacing = None, step = None, interpolate = 2, mask = None, fshape = None, dtype = None, return_centers = False):
  """Calculate local histograms on a sub-grid, apply a scalar valued function and resmaple to original image shape.
  
  Arguments
  ---------
  source : array
    The source to process.
  function : function
    Function to apply to the linear array of the local source data.
    If the function does not return a scalar, fshape has to be given.
  selem : tuple or array or None
    The structural element to use to extract the local image data.
    If tuple, use a rectangular region of this shape. If array, the array
    is assumed to be bool and acts as a local mask around the center point.
  spacing : tuple or None
    The spacing between sample points. If None, use shape of selem.
  step : tuple of int or None
    If tuple, subsample the local region by these step. Note that the
    selem is applied after this subsampling.
  interpolate : int or None
    If int, resample the result back to the original source shape using this
    order of interpolation. If None, return the results on the sub-grid.
  mask : array or None
    Optional mask to use.
  fshape : tuple or None
    If tuple, this is the shape of the function output. 
    If None assumed to be (1,).
  dtype : dtype or None
    Optional data type for the result.
  return_centers : bool
    If True, additionaly return the centers of the sampling.
    
  Returns
  -------
  local : array
    The reuslt of applying the function to the local samples.
  cetners : array
    Optional cttners of the sampling.
  """
  
  if spacing is None:
    spacing = selem;
  shape = source.shape;
  ndim = len(shape);
  
  if step is None:
    step = (None,) * ndim
    
  if len(spacing) != ndim or len(step) != ndim:
    raise ValueError('Dimension mismatch in the parameters!')  
    
  #histogram centers
  n_centers = tuple(s//h for s,h in zip(shape, spacing))
  left = tuple((s - (n-1) * h)//2  for s,n,h in zip(shape, n_centers, spacing));
  
  #center points
  centers = np.array(np.meshgrid(*[range(l, s, h) for l,s,h in zip(left, shape, spacing)], indexing = 'ij'));
  #centers = np.reshape(np.moveaxis(centers, 0, -1),(-1,len(shape)));               
  centers = np.moveaxis(centers, 0, -1)                                          
  
  #create result
  rshape = (1,) if fshape is None else fshape;
  rdtype = source.dtype if dtype is None else dtype;
  results = np.zeros(n_centers + rshape, dtype = rdtype);
  
  #calculate function
  centers_flat = np.reshape(centers, (-1,ndim));
  results_flat = np.reshape(results, (-1,) + rshape);
  
  #structuring element
  if isinstance(selem, np.ndarray):
    selem_shape = selem.shape
  else:
    selem_shape = selem;
    selem = None
  
  hshape_left = tuple(h//2 for h in selem_shape);           
  hshape_right = tuple(h - l for h,l in zip(selem_shape, hshape_left));
  
  for result, center in zip(results_flat,centers_flat):
    sl = tuple(slice(max(0,c-l), min(c+r,s), d) for c,l,r,s,d in zip(center, hshape_left, hshape_right, shape, step));
    if selem is None:
      if mask is not None:
        data = source[sl][mask[sl]];
      else:
        data = source[sl].flatten();
    else:
      slm = tuple(slice(None if c-l >= 0 else min(l-c,m), None if c+r <= s else min(m - (c + r - s), m), d) for c,l,r,s,d,m in zip(center, hshape_left, hshape_right, shape, step, selem_shape));
      data = source[sl];
      if mask is not None:
        data = data[np.logical_and(mask[sl], selem[slm])]
      else:
        data = data[selem[slm]];
      
    #print result.shape, data.shape, function(data)
    result[:] = function(data);
  
  #resample
  try:
    if interpolate:
      res_shape = results.shape[:len(shape)];
      zoom = tuple(float(s) / float(r) for s,r in zip(shape, res_shape));
      results_flat = np.reshape(results, res_shape + (-1,));  
      results_flat = np.moveaxis(results_flat, -1, 0);
      full = np.zeros(shape + rshape, dtype = results.dtype);
      full_flat = np.reshape(full, shape + (-1,));     
      full_flat = np.moveaxis(full_flat, -1, 0); 
      #print results_flat.shape, full_flat.shape
      for r,f in zip(results_flat, full_flat):
        f[:] = ndi.zoom(r, zoom=zoom, order = interpolate);   
      results = full;
    
    if fshape is None: 
      results.shape = results.shape[:-1];
  
    if return_centers:
      return results, centers
    else:
      return results
  except:
    return 

def local_histogram(source, max_bin = 2**12, selem = (50,50), spacing = None, step = None, interpolate = None, mask = None, dtype = None, return_centers = False):
  """Calculate local histograms on a sub-grid.
  
  Arguments
  ---------
  source : array
    The source to process.
  selem : tuple or array or None
    The structural element to use to extract the local image data.
    If tuple, use a rectangular region of this shape. If array, the array
    is assumed to be bool and acts as a local mask around the center point.
  spacing : tuple or None
    The spacing between sample points. If None, use shape of selem.
  step : tuple of int or None
    If tuple, subsample the local region by these step. Note that the
    selem is applied after this subsampling.
  interpolate : int or None
    If int, resample the result back to the original source shape using this
    order of interpolation. If None, return the results on the sub-grid.
  mask : array or None
    Optional mask to use.
  max_bin : int
    Maximal bin value to account for.
  return_centers : bool
    If True, additionaly return the centers of the sampling.
    
  Returns
  -------
  histograms : array
    The local histograms.
  cetners : array
    Optional centers of the sampling.
    
  Note
  ----
  For speed, this function works only for uint sources as the histogram is 
  calculated directly via the source values. The source values should be 
  smaller than max_bin.
  """
  
  def _hist(data):
    data, counts = np.unique(data, return_counts=True);
    histogram = np.zeros(max_bin, dtype=int);
    histogram[data] = counts;
    return histogram;
  
  return apply_local_function(source, selem=selem, spacing=spacing, step=step, interpolate=interpolate, mask=mask, dtype=dtype, return_centers=return_centers,
                             function=_hist, fshape = (max_bin,));
    


def local_percentile(source, percentile, selem = (50,50), spacing = None, step = None, interpolate = 1, mask = None, dtype = None, return_centers = False):
  """Calculate local percentile.
  
  Arguments
  ---------
  source : array
    The source to process.
  percentile : float or array
    The percentile(s) to estimate locally.
  selem : tuple or array or None
    The structural element to use to extract the local image data.
    If tuple, use a rectangular region of this shape. If array, the array
    is assumed to be bool and acts as a local mask around the center point.
  spacing : tuple or None
    The spacing between sample points. If None, use shape of selem.
  step : tuple of int or None
    If tuple, subsample the local region by these step. Note that the
    selem is applied after this subsampling.
  interpolate : int or None
    If int, resample the result back to the original source shape using this
    order of interpolation. If None, return the results on the sub-grid.
  mask : array or None
    Optional mask to use.
  return_centers : bool
    If True, additionaly return the centers of the sampling.
    
  Returns
  -------
  percentiles : array
    The local percentiles.
  cetners : array
    Optional centers of the sampling.
  """
  if isinstance(percentile, (tuple, list)):
    percentile = np.array([100*p for p in percentile]);
    fshape = (len(percentile),)
    def _percentile(data):
      if len(data) == 0:
        return np.array((0,) * len(percentile));
      return np.percentile(data, percentile, axis = None);
  
  else:
    percentile = 100 * percentile;
    fshape = None;
    def _percentile(data):
      if len(data) == 0:
        return 0;
      return np.percentile(data, percentile, axis = None);
  
  return apply_local_function(source, selem=selem, spacing=spacing, step=step, interpolate=interpolate, mask=mask, dtype=dtype, return_centers=return_centers,
                              function=_percentile, fshape=fshape);


###############################################################################
### Tests
###############################################################################                         
                              
# def _test():
#   """Tests."""
#   import numpy as np
#   import ClearMap.Visualization.Plot3d as p3d

#   import ClearMap.ImageProcessing.LocalStatistics as ls     
#   from importlib import reload
#   reload(ls)                
                      
#   source = np.random.rand(100,200,150) + np.arange(100)[:,None,None];
#   p = ls.local_percentile(source, percentile=0.5, selem=(30,30,30), interpolate=1);
#   p3d.plot([source, p])
  
                           
                              
                              
                              
                              
#def apply_histogram_function(histograms, function, shape = None, interpolation = 2):
#
#  hist_shape = histograms.shape[:-1];
#  max_bin = histograms.shape[-1];
#  result = np.zeros(np.prod(hist_shape), dtype = float);
#  histograms_flat = np.reshape(histograms, (-1,max_bin));
#  for i,h in enumerate(histograms_flat):
#    result[i] = function(h);
#  result.shape = hist_shape;
#  
#  if shape is not None:
#    zoom = tuple(float(s) / float(h) for s,h in zip(shape, hist_shape))
#    result = ndi.zoom(result, zoom=zoom, order = interpolation);
#  
#  return result;
