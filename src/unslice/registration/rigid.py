# Module for rigid registration using elastix 

from .. import IO as io 
import os 
import subprocess 
import numpy as np
from ..utils import get_chunk_coords, upsample_zarr   
import matplotlib.pyplot as plt 
import multiprocessing as mp 
# from skimage.transform import rotate 
from scipy.ndimage import rotate 
from functools import partial 
import zarr 
import time
from tqdm import tqdm 

temp_dir = 'temp/rigid_tiffs'
#shell = True # leave as False, UNLESS running in jupyter notebook 

def _rescale_params(tfm_name, tfm_new_name, scale_factor, fixed_img_size, transform_type='rigid'):
    '''
    Modify the Elastix parameters to work for the original image resolution 
    
    Inputs:
    tfm_name = filepath of the transform parameters for downsampled image 
    tfm_new_name = filepath of the new transform parameters for real image 
    scale_factor = how much the downsampled image was scaled 
    fixed_img_size = (x,y) size of the fixed image 
    transform_type = str, can be 'rigid', 'affine', or 'bspline'. This function will rescale 
                     parameters based on which is specified. 
    
    Outputs:
    the new transforms, along with saved new transform parameter file for non-downsampled image 
    '''
    
    transform_file = open(tfm_name, 'r')
    # the parameters are located on the third line - scale up the translation parameters 
    lines = transform_file.readlines()
    transform_line = lines[2]
    transform_line = transform_line.split()
    if transform_type == 'rigid':
        trans_x = float(transform_line[2]) * scale_factor
        trans_y = float(transform_line[3][:-1]) * scale_factor 
        
        # the center of rotation also needs to be changed 
        center_line = lines[19]
        center_line = center_line.split()

        ## Potentially we need to change this in order to make it consistent with the fixed image size
        # center_x = float(center_line[1]) * scale_factor 
        # center_y = float(center_line[2][:-1]) * scale_factor 
        center_x = fixed_img_size[0] * 0.5 
        center_y = fixed_img_size[1] * 0.5 
        
        # Write to a new parameter file 
        new_file = open(tfm_new_name, 'w')
        i = 0 
        for line in lines:
            if i == 2:
                # Change the translation parameters 
                rot = float(transform_line[1])
                new_file.write('(TransformParameters ' + str(rot) + ' ' + str(trans_x) + ' ' + str(trans_y) + ')\n')
            elif i == 11:
                # Change the size of the image 
                new_file.write('(Size ' + str(fixed_img_size[0]) + ' ' + str(fixed_img_size[1]) + ')\n')
            elif i == 19: 
                # change the center of rotation 
                new_file.write('(CenterOfRotationPoint ' + str(center_x) + ' ' + str(center_y) + ')\n')
            elif i == 28: 
                new_file.write('(ResultImageFormat "tiff")\n')
            elif i == 29:
                new_file.write('(ResultImagePixelType "short")\n')
            else:
                new_file.write(line) 
            i += 1
    elif transform_type == 'bspline':
        # Get rid of final parantheses, and get rid of the first string 
        transform_line[-1] = transform_line[-1][:-1]
        transform_line = transform_line[1:]
        t_line = [float(number) for number in transform_line]
        
        # Other parameters that need to be changed 
        gridspacing_x = float(lines[21].split()[1])
        gridspacing_y = float(lines[21].split()[2][:-1])
        gridorigin_x = float(lines[22].split()[1])
        gridorigin_y = float(lines[22].split()[2][:-1])
        # Write to a new parameter file 
        new_file = open(tfm_new_name, 'w')
        i = 0 
        for line in lines:
            if i == 2:
                # Change the B Spline deformation field 
                str_to_write = '(TransformParameters'
                for param in t_line:
                    str_to_write += ' ' + str(param*scale_factor)
                str_to_write += ')\n'
                new_file.write(str_to_write)
            elif i == 11:
                # Change the size of the image 
                new_file.write('(Size ' + str(fixed_img_size[0]) + ' ' + str(fixed_img_size[1]) + ')\n')
            elif i == 21: 
                new_file.write('(GridSpacing ' + str(gridspacing_x*scale_factor) + ' ' + str(gridspacing_y*scale_factor) + ')\n')
            elif i == 22:
                new_file.write('(GridOrigin ' + str(gridorigin_x*scale_factor) + ' ' + str(gridorigin_y*scale_factor) + ')\n')
            elif i == 28: 
                new_file.write('(ResultImageFormat "tiff")\n')
            elif i == 29:
                new_file.write('(ResultImagePixelType "short")\n')
            else:
                new_file.write(line) 
            i += 1
    elif transform_type == 'affine':
        pass
    else:
        raise RuntimeError('Invalid type of transform.')

def nonrigid3d_register(elastix_path, parameter_path, fixed_mip_path, moving_mip_path, 
                   fixed_img_size, resample_parameter=(1,1,1), num_workers=8,
                   moving_zarr_path=None, moving_surf_zarr_path=None,
                   moving_warped_zarr_path=None, moving_warped_surf_zarr_path=None,
                   align_images=True, shell=False, use_sitk=False):
    '''
    Perform nonrigid surface registration

    Inputs:
    elastix_path - str, the path to which we store the transform parameters 
    parameter_path - str, the path to a file that has the rigid registration parameters in it 
                     if a list of str, then we perform registration in sequence 
    fixed_zarr_path - path to the fixed 2D image 
    moving_zarr_path - path to the moving 2D image 
    fixed_img_size - tuple (x,y), the size of the undownsampled fixed image 
    resample_parameter - int, the number of times each dimension x,y have been downsampled
    num_workers - int, number of parallel processes to run 
    moving_fullres_zarr_path - str, if not None, the path to the upsampled sequence of images (potentially zarr) to transform 
    moving_surf_zarr_path - str, if not None, the path to the sequence of images 
    moving_warped_zarr_path - str, if not None, the sink path to the warped image 
    moving_warped_surf_zarr_path - str, if not None, the sink path to warped surface 
    align_images - bool, if True, will go through the entire process. if False, will use existing transform.parameters to align. 
    shell - bool, if True, will spawn a shell. Only use shell=True for jupyter notebooks.
    use_sitk - bool, if True, then use SimpleElastix, else use command line Elastix. default: False
    '''

    if use_sitk is None or use_sitk == True: 
        try:
            import SimpleITK as sitk 
            elastixImageFilter = sitk.ElastixImageFilter() 
            use_sitk = True 
        except:
            use_sitk = False 
            print('SimpleElastix not installed (properly), now running in command line Elastix mode.')

    if not os.path.exists(elastix_path):
        os.makedirs(elastix_path)

    if not use_sitk: 
        if align_images:
            # First we need to make the directory with all of the 

            cmd2run = 'elastix -f ' + fixed_mip_path + ' -m ' + moving_mip_path + ' -out ' + elastix_path
            if type(parameter_path) == str:
                cmd2run = cmd2run + ' -p ' + parameter_path 
            elif type(parameter_path) == list:
                for parameters in parameter_path:
                    cmd2run += ' -p ' + parameters 
            # if use_mask:
                # cmd2run = cmd2run + ' -fMask ' + segmentedFix + ' -mMask ' + segmentedMove 
            subprocess.run(cmd2run, shell=shell) 
            
            check = input('Are you satisfied with the registration result (elastix_path/result.tiff)? ([y]/n)') 
            if check == 'n' or check == 'N':
                return 
            
            transform_file = os.path.join(elastix_path,'TransformParameters.0.txt')
            if resample_parameter != 1:
                _rescale_params(transform_file,
                                      os.path.join(elastix_path,'ScaledTransformParameters.0.txt'),
                                      resample_parameter, fixed_img_size, transform_type='rigid') 
                transform_file = os.path.join(elastix_path,'ScaledTransformParameters.0.txt')
            
                # TODO: write a function ot rescale the B Spline parameter file. 
                if os.path.isfile(os.path.join(elastix_path, 'TransformParameters.1.txt')):
                    # function that can rescale the B Spline parameters. 
                    # pass 
                    _rescale_params(os.path.join(elastix_path, 'TransformParameters.1.txt'),
                                            os.path.join(elastix_path, 'ScaledTransformParameters.1.txt'),
                                            resample_parameter, fixed_img_size, transform_type='bspline')
                    transform_file_nonrigid = os.path.join(elastix_path, 'ScaledTransformParameters.1.txt')
        else:
            transform_file = os.path.join(elastix_path,'ScaledTransformParameters.0.txt')

        if moving_warped_zarr_path is not None: 
            _rigid_transform_stack(transform_file, fixed_img_size, moving_zarr_path, 
                                    moving_warped_zarr_path, num_workers=num_workers, shell=shell)
                                    
        
                      
        # Then warp the surface as well 
        if moving_warped_surf_zarr_path is not None:
            # Resample the surface back to the full resolution (if not already at full resolution) 
            if resample_parameter != 1:
                temp_upsample_mask_path = os.path.join(temp_dir, 'temp_upsampled_mask.zarr')
                move_shape = zarr.open(moving_zarr_path,mode='r').shape
                upsample_zarr(moving_surf_zarr_path, temp_upsample_mask_path, 2*(resample_parameter,)+(1,), final_image_size=move_shape, 
                          num_workers=num_workers, is_mask=True)
            else: 
                temp_upsample_mask_path = moving_surf_zarr_path 
            
            # Warp the surface 
            _rigid_transform_stack(transform_file, fixed_img_size, temp_upsample_mask_path, 
                                    moving_warped_surf_zarr_path, num_workers=num_workers, shell=shell)
    else:
        # use SimpleElastix 
        # This part is not yet written because subprocesses work fine. 
        elastixImageFilter.SetMovingImage(sitk.ReadImage(moving_mip_path))
        elastixImageFilter.SetFixedImage(sitk.ReadImage(fixed_img_path))

        # Parameter map for rigid registration 
        parameterMap = sitk.GetDefaultParameterMap('rigid')
        parameterMap['Number']
        pass       

    pass

        
def rigid_register(elastix_path, parameter_path, fixed_mip_path, moving_mip_path, 
                   fixed_img_size, resample_parameter=1, num_workers=8,
                   moving_zarr_path=None, moving_surf_zarr_path=None,
                   moving_warped_zarr_path=None, moving_warped_surf_zarr_path=None,
                   align_images=True, shell=False, use_sitk=False):
    '''
    Perform rigid registration of images  
    
    Inputs:
    elastix_path - str, the path to which we store the transform parameters 
    parameter_path - str, the path to a file that has the rigid registration parameters in it 
                     if a list of str, then we perform registration in sequence 
    fixed_mip_path - path to the fixed 2D image 
    moving_mip_path - path to the moving 2D image 
    fixed_img_size - tuple (x,y), the size of the undownsampled fixed image 
    resample_parameter - int, the number of times each dimension x,y have been downsampled
    num_workers - int, number of parallel processes to run 
    moving_zarr_path - str, if not None, the path to the sequence of images (potentially zarr) to transform 
    moving_surf_zarr_path - str, if not None, the path to the sequence of images 
    moving_warped_zarr_path - str, if not None, the sink path to the warped image 
    moving_warped_surf_zarr_path - str, if not None, the sink path to warped surface 
    align_images - bool, if True, will go through the entire process. if False, will use existing transform.parameters to align. 
    shell - bool, if True, will spawn a shell. Only use shell=True for jupyter notebooks.
    use_sitk - bool, if True, then use SimpleElastix, else use command line Elastix. default: False
    '''
    
    if use_sitk is None or use_sitk == True: 
        try:
            import SimpleITK as sitk 
            elastixImageFilter = sitk.ElastixImageFilter() 
            use_sitk = True 
        except:
            use_sitk = False 
            print('SimpleElastix not installed (properly), now running in command line Elastix mode.')


    if not os.path.exists(elastix_path):
        os.makedirs(elastix_path) 
    
    if not use_sitk: 
        if align_images:
            cmd2run = 'elastix -f ' + fixed_mip_path + ' -m ' + moving_mip_path + ' -out ' + elastix_path
            if type(parameter_path) == str:
                cmd2run = cmd2run + ' -p ' + parameter_path 
            elif type(parameter_path) == list:
                for parameters in parameter_path:
                    cmd2run += ' -p ' + parameters 
            # if use_mask:
                # cmd2run = cmd2run + ' -fMask ' + segmentedFix + ' -mMask ' + segmentedMove 
            subprocess.run(cmd2run, shell=shell) 
            
            check = input('Are you satisfied with the registration result (elastix_path/result.tiff)? ([y]/n)') 
            if check == 'n' or check == 'N':
                return 
            
            transform_file = os.path.join(elastix_path,'TransformParameters.0.txt')
            if resample_parameter != 1:
                _rescale_params(transform_file,
                                      os.path.join(elastix_path,'ScaledTransformParameters.0.txt'),
                                      resample_parameter, fixed_img_size, transform_type='rigid') 
                transform_file = os.path.join(elastix_path,'ScaledTransformParameters.0.txt')
            
                # TODO: write a function ot rescale the B Spline parameter file. 
                if os.path.isfile(os.path.join(elastix_path, 'TransformParameters.1.txt')):
                    # function that can rescale the B Spline parameters. 
                    # pass 
                    _rescale_params(os.path.join(elastix_path, 'TransformParameters.1.txt'),
                                            os.path.join(elastix_path, 'ScaledTransformParameters.1.txt'),
                                            resample_parameter, fixed_img_size, transform_type='bspline')
                    transform_file_nonrigid = os.path.join(elastix_path, 'ScaledTransformParameters.1.txt')
        else:
            transform_file = os.path.join(elastix_path,'ScaledTransformParameters.0.txt')

        if moving_warped_zarr_path is not None: 
            _rigid_transform_stack(transform_file, fixed_img_size, moving_zarr_path, 
                                    moving_warped_zarr_path, num_workers=num_workers, shell=shell)
                                    
        
                      
        # Then warp the surface as well 
        if moving_warped_surf_zarr_path is not None:
            # Resample the surface back to the full resolution (if not already at full resolution) 
            if resample_parameter != 1:
                temp_upsample_mask_path = os.path.join(temp_dir, 'temp_upsampled_mask.zarr')
                move_shape = zarr.open(moving_zarr_path,mode='r').shape
                upsample_zarr(moving_surf_zarr_path, temp_upsample_mask_path, 2*(resample_parameter,)+(1,), final_image_size=move_shape, 
                          num_workers=num_workers, is_mask=True)
            else: 
                temp_upsample_mask_path = moving_surf_zarr_path 
            
            # Warp the surface 
            _rigid_transform_stack(transform_file, fixed_img_size, temp_upsample_mask_path, 
                                    moving_warped_surf_zarr_path, num_workers=num_workers, shell=shell)
    else:
        # use SimpleElastix 
        # This part is not yet written because subprocesses work fine. 
        elastixImageFilter.SetMovingImage(sitk.ReadImage(moving_mip_path))
        elastixImageFilter.SetFixedImage(sitk.ReadImage(fixed_img_path))

        # Parameter map for rigid registration 
        parameterMap = sitk.GetDefaultParameterMap('rigid')
        parameterMap['Number']
        pass                              
     
    
    
def _rigid_transform_stack(transform_file, fixed_img_size, moving_zarr_path, moving_warped_zarr_path, num_workers=None, shell=False):
    '''
    Apply a 2D rigid transform to a stack 
    '''
    global temp_dir 
    start = time.time()
    try:
        og = zarr.open(moving_zarr_path, mode='r')
        num_slices = og.shape[2] 
        dtyp = 'zarr'
    except:
        num_slices = len(os.listdir(moving_zarr_path))
        dtyp = 'tif'
        
    warped = zarr.open(moving_warped_zarr_path, mode='w',
                       shape=(*fixed_img_size,og.shape[2]),
                       chunks=(*fixed_img_size,og.chunks[2]),
                       dtype=np.uint16) 
    if num_workers is None:
        for z in tqdm(range(num_slices)):
            if dtyp == 'zarr':
                # Save a temp tiff file 
                filename = os.path.join(temp_dir,'temp.tif')
                if not os.path.isdir(temp_dir):
                    os.mkdir(temp_dir)
                io.writeData(filename,og[:,:,z].astype('uint16'))
            else:
                filename = os.path.join(moving_zarr_path, os.listdir(moving_zarr_path)[z])
            cmd2run = 'transformix -in ' + filename + ' -out ' + temp_dir 
            cmd2run = cmd2run + ' -tp ' + transform_file 
            subprocess.run(cmd2run, shell=shell) 
            
            # Saves as "result.tiff" - convert back to zarr 
            res = io.readData(os.path.join(temp_dir,'result.tiff'))
            warped[:,:,z] =  res.clip(min=0).astype('uint16')
            #print('Finished processing slice %d in %f seconds'%(z,time.time()-start))
    else:
        return 
       

####### Functions for doing some basic transforms of zarr files ########
def rotate_2d_zarr(source_zarr_path, sink_zarr_path, angle=180, num_workers=8):
    '''
    Rotates an image by some angle.   

    Inputs:
    source_zarr_path - str
    sink_zarr_path - str 
    angle - float, counterclockwise angle of rotation
    num_workers - int 
    '''

    img = zarr.open(source_zarr_path, mode='r')

    # Determine the shape of the rotated image 
    test_img = rotate_2d(img[:,:,0], angle)
    rotated_img = zarr.open(sink_zarr_path, mode='w',
        shape = (*test_img.shape[:2],img.shape[2]), 
        chunks=(*test_img.shape[:2],img.chunks[2]), 
        dtype=img.dtype)

    coord_ranges = get_chunk_coords(img.shape, (*img.shape[:2],img.chunks[2]))
    if num_workers > 1:
        p = mp.Pool(num_workers)
        f = partial(_rotate_2d_zarr_serial, img, rotated_img, angle)
        list(tqdm(p.imap(f, coord_ranges), total=len(coord_ranges)))
        p.close()
        p.join()
    else:
        for coords in tqdm(coord_ranges):
            _rotate_2d_zarr_serial(img, rotated_img, angle, coords)


def _rotate_2d_zarr_serial(source_zarr, sink_zarr, angle, coords):
    '''
    Kernel for parallelization
    '''

    start = time.time()
    _,_,zr = coords 
    img_slices = source_zarr[:,:,zr[0]:zr[1]]
    rotated_img_slice = np.zeros((*sink_zarr.shape[:2],zr[1]-zr[0]),dtype=sink_zarr.dtype)
    
    for z in np.arange(0,zr[1]-zr[0]):
        rotated_img_slice[:,:,z] = rotate_2d(img_slices[:,:,z], angle).astype('int')
        
    sink_zarr[:,:,zr[0]:zr[1]] = rotated_img_slice 
    print("%d-%d slices done in %f seconds"%(zr[0],zr[1]-1,time.time()-start))

def rotate_2d(img, angle):
    '''
    Rotates a numpy array by angle.

    Inputs:
    img - 2D array 
    angle - float, the number of degrees for counter-clockwise 

    Outputs:
    rotated_img 
    '''
    if angle != 180:
        # only resize the image if we aren't rotating by 180 
        resize = True 
    else:
        resize = False 

    return rotate(img.astype('float32'), angle, resize=resize)




def flip_zarr(source_zarr_path, sink_zarr_path, axis, num_workers=8):
    '''
    Flips the image by axis 0 (vertical), 1 (horizontal), or 2 (z) 

    Inputs:
    source_zarr_path - str
    sink_zarr_path - str 
    axis - int, 0: vertical, 1: horizontal, 2: z
    num_workers - int 
    '''

    img = zarr.open(source_zarr_path, mode='r')

    # Determine the shape of the rotated image 
    flipped_img = zarr.open(sink_zarr_path, mode='w',
        shape = img.shape, chunks=img.chunks, dtype=img.dtype)

    coord_ranges = get_chunk_coords(img.shape, img.chunks)
    if num_workers > 1:
        p = mp.Pool(num_workers)
        f = partial(flip_image, img, flipped_img, axis)
        list(tqdm(p.imap(f, coord_ranges), total=len(coord_ranges)))
        p.close()
        p.join()
    else:
        for coords in tqdm(coord_ranges):
            flip_image(img, flipped_img, axis, coords)



def flip_image(source_zarr, sink_zarr, axis, coords):
    _,_,zr = coords 
    if zr[1] > sink_zarr.shape[2]:
        zr[1] = sink_zarr.shape[2]
    img_slices = source_zarr[:,:,zr[0]:zr[1]]
    # flipped_img_slice = np.zeros((*sink_zarr.shape[:2],zr[1]-zr[0]),dtype=sink_zarr.dtype)
    flipped_img_slice = np.zeros(img_slices.shape,dtype=sink_zarr.dtype)
    # for z in np.arange(0,zr[1]-zr[0]):
    for z in np.arange(0,img_slices.shape[2]):
        if axis == 0 or axis == 1:
            flipped_img_slice[:,:,z] = np.flip(img_slices[:,:,z], axis)
        elif axis == 2:
            # print(zr, flipped_img_slice.shape, img_slices.shape)
            flipped_img_slice[:,:,z] = img_slices[:,:,img_slices.shape[2]-z-1]
    if axis == 0 or axis == 1:
        sink_zarr[:,:,zr[0]:zr[1]] = flipped_img_slice 
    elif axis == 2:
        sink_zarr[:,:,sink_zarr.shape[2]-zr[1]:sink_zarr.shape[2]-zr[0]] = flipped_img_slice 


### Try to do everything with SimpleElastix: allow for nonrigid downsample dregistration 



######## More functionalities for rigid and affine trnasforms ################

def rigid_transform(M, b, pts):
    return np.matmul(M,pts.T).T + b

def define_rigid_transform(Scx=1.0, Scy=1.0, Scz=1.0, x_angle=0.0, y_angle=0.0, z_angle=0.0, x_trans=0.0, y_trans=0.0, z_trans=0.0):
    
    z_angle *= np.pi/180
    y_angle *= np.pi/180
    x_angle *= np.pi/180 

    cx = np.cos(x_angle); sx = np.sin(x_angle)
    cy = np.cos(y_angle); sy = np.sin(y_angle)
    cz = np.cos(z_angle); sz = np.sin(z_angle)
    x_theta = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    y_theta = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    z_theta = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    R = np.matmul(np.matmul(x_theta,y_theta),z_theta)
    
    # Account for scaling
    scale_matrix = np.array([[Scx,0,0],[0,Scy,0],[0,0,Scz]])
    R = np.matmul(R,scale_matrix)

    b = np.array([x_trans, y_trans, z_trans])
    
    return R,b

def get_inverse_rigid_transform(M,b):
    Minv = np.zeros((4,4))
    Minv[:3,:3] = M
    Minv[:3,3] = b
    Minv[3,3] = 1
    
    Minv = np.linalg.inv(Minv)
    binv = Minv[:3,3]
    Rinv = Minv[:3,:3]
    
    return Rinv,binv

def manual_rigid_transform(og_points, ref_points, Scx, Scy, Scz, x_angle, y_angle, z_angle, 
                            x_trans, y_trans, z_trans, plot=True, return_matrices=False):
    # Compute the transformation matrices 
    R,b = define_rigid_transform(Scx, Scy, Scz, x_angle, y_angle, z_angle, 
                            x_trans, y_trans, z_trans)

    # Perform the transform
    new_points = np.zeros(og_points.shape)
    for i in range(len(og_points)):
        newpoint = np.matmul(R,og_points[i])+b
        new_points[i] = newpoint

    # Check transformation
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.scatter(ref_points[:,0],ref_points[:,1],ref_points[:,2],antialiased=True, alpha=0.5)
        ax.scatter(og_points[:,0],og_points[:,1],og_points[:,2],antialiased=True,alpha=0.5)
        ax.scatter(new_points[:,0],new_points[:,1],new_points[:,2],antialiased=True,alpha=0.5,color='g')

    if return_matrices:
        return new_points, R, b
    else:
        return new_points



def rigid_transform_3D(A, B):
    '''
    Input: expects 3xN matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector

    Adapted from https://github.com/nghiaho12/rigid_transform_3D 
    '''
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3 and num_rows != 2:
        raise Exception(f"matrix A is not 3xN or 2xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3 and num_rows != 2:
        raise Exception(f"matrix B is not 3xN or 2xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1 or 2x1 
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

