import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import zarr 
import itertools

# 
def compute_new_img_shape(old_img_size, R, b):
    '''
    Compute the image size after an affine transform 

    Also returns the coordinates needed to be added 
    '''
    lists = []
    for a in old_img_size:
        lists.append([0,a])

    coords = []
    for r in itertools.product(*tuple(lists)):
        newpt = np.matmul(R,np.array(r)) + b
        coords.append(newpt)
    coords = np.array(coords)
    
    # Find the max range 
    min_coords = np.round(np.min(coords, axis=0))
    max_coords = np.round(np.max(coords, axis=0))
    # return max_coords-min_coords, min_coords 
    return max_coords.astype('int')


# Function to be added
def get_z_translation(surf_zarr, slab, z_orientation='bottom'):
    '''
    surf_zarr: zarr, the (downsampled) surface zarr from which we are extracting information
    z_orientation: 'top' or 'bottom'; if 'bottom', then the surface is located at z=0 (default: 'bottom')
    slab: 'top' or 'bottom'; if 'top', then the returned z_translation is negative. If 'bottom', then the returned z_translation is positive. 
    
    This function already assumes that coordinates have been transformed to be correctly oriented, but not the zarr slabs.
    Also assumes that "bottom" refers to z=0, and top refers to the end 
    '''

    if z_orientation == 'top':
        zrange = np.flip(np.arange(surf_zarr.shape[2]))
    else:
        zrange = np.arange(surf_zarr.shape[2])
    for i in zrange:
        # The first optical section with any segmented surface pixels 
        if np.sum(surf_zarr[:,:,i]) > 0:
            if z_orientation == 'top':
                translation = surf_zarr.shape[2]-1-i
            else:
                translation = i 
            break
    if slab == 'top':
        return -translation
    else:
        return translation 
        

def fit_surface(surf_zarr, num_points_grid, num_samples, order, zflip=True, zadd=False, num_bottom_slices=None, plot=False):
    '''
    surf_zarr: EITHER zarr containing the surface (probably downsampled data),
               OR numpy array (of shape (n,3)) containing point coordinates of already sampled data
    num_point_grid: int, number of points in x and y directions of grid for fitting 
    num_samples: int, number of coordinates in actual surface to sample for fitting
    order: int (1 or 2), if 1, fit a linear plane; if 2, fit a quadratic curve 
    zflip: bool, if True, then flips the z of the coordinates
    zadd: bool, if True, add num_bottom_slices to the coordinates
    num_bottom_slices: if not None and zflip is True, then is the full number of bottom slices in full image
    plot: bool, if True, will plot the sampled surface and the fitted plane/curve on top 
    '''

    if type(surf_zarr) == zarr.core.Array:
        # If it's a zarr, then sample points on the surface for fitting. 
        data = _sample_surface(surf_zarr, num_samples)
    elif type(surf_zarr) == np.ndarray:
        if surf_zarr.shape[1] == 3: # only accept 3D plane fitting 
            data = surf_zarr.copy() 

    if zflip:
        if num_bottom_slices is None:
            num_bottom_slices = surf_zarr.shape[2]
        data[:,2] = num_bottom_slices - data[:,2] - 1
    elif zadd:
        if num_bottom_slices is None:
            print('No num_bottom_slices given, using surf_zarr')
        data[:,2] = num_bottom_slices + data[:,2]

    if type(surf_zarr) == zarr.core.Array:
        X,Y = np.meshgrid(surf_zarr.shape[0]*np.linspace(0.0, 1.0, num_points_grid), 
                          surf_zarr.shape[1]*np.linspace(0.0, 1.0, num_points_grid))
    elif type(surf_zarr) == np.ndarray and surf_zarr.shape[1] == 3:
        X,Y = np.meshgrid(data[:,0].max()*np.linspace(0.0,1.0,num_points_grid),
                        data[:,1].max()*np.linspace(0.0,1.0,num_points_grid))

    #XX = X.flatten()
    #YY = Y.flatten()
    data = data.astype('int32')
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,res,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients, residues
        
        ## evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]
    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), np.square(data[:,:2])]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
        
        # evaluate it on a grid
        # Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
        Z = C[0] + C[1]*X + C[2]*Y + C[3]*X*Y + C[4]*X*X + C[5]*Y*Y 
    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50, alpha=0.1)
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('equal')
        ax.axis('tight')
        plt.show()
        
    return C, X, Y, Z

def compute_fit_surface_residuals(pts, C):
    '''
    Computes the residuals matrix for all of the points based on the 
    fit plane coefficients C
    '''
    Z = C[0]*pts[:,0] + C[1]*pts[:,1] + C[2]
    return np.abs(Z - pts[:,2])

def _sample_surface(surf_zarr, num_samples):
    # helper function for sampling surface coordinates for fit_surface 
    surf = surf_zarr[:]
    coords = np.argwhere(surf)
    indices = np.random.choice(coords.shape[0], size=num_samples, replace=False)
    return coords[indices]

def grid_sample_surface(surf_zarr_path, downsample_factor=(1,1,1), num_random_samples=None, grid_size=None):
    '''
    sample a surface zarr (mask representing the surface of an image)
    surf_zarr_path - path to the zarr (downsampled)
    downsample_factor - how much the zarr is downsampled from the real image
    num_random_samples - if not None, number of points ot randomly sample. Else, we use a regular grid
    grid_size - 2-tuple, if not None and random_sample is False, then we sample a grid of size grid_size
    
    returns list of coordinates that are on the surface 
    '''
    z = zarr.open(surf_zarr_path, mode='r')
    if num_random_samples is not None:
        coords = _sample_surface(z, num_random_samples)
    else:
        thinsurf = z[:]
        surfcoords = np.argwhere(thinsurf)
        xr,yr = thinsurf.shape[:2]

        # make grid
        xs = np.round(np.linspace(0,xr,grid_size[0]))
        ys = np.round(np.linspace(0,yr,grid_size[1]))
        X,Y = np.meshgrid(xs,ys)
        coordsxy = np.array([X.flatten(),Y.flatten()]).T.astype('int')

        # match the sets to find where we have detected surface 
        coordsxy_set = set([tuple(v) for v in coordsxy])
        surfxy_set = set([tuple(v) for v in surfcoords[:,:2]])
        intersectxy = surfxy_set.intersection(coordsxy_set)

        sampled_surf_pts = []
        for coord in intersectxy:
            coord_toadd = surfcoords[(surfcoords[:,0]==coord[0]) * (surfcoords[:,1]==coord[1])]
            sampled_surf_pts.append(coord_toadd)
        coords = np.array(sampled_surf_pts).squeeze()

    return coords

    

def plane_rigid_transform_matrix(moving_plane,fixed_plane):
    # moving_plane = [A1,B1,C1]
    # fixed_plane = [A2,B2,C2]
    # in the format z = A1*x + B1*y + C1, to be transformed to be z = A2*x + B2*y + C2
    
    # We first rotate it using formulas from geometry
    A1,B1,C1 = moving_plane 
    A2,B2,C2 = fixed_plane 
    n1 = np.array([A1,B1,-1])
    n1_norm = n1/np.linalg.norm(n1)
    n2 = np.array([A2,B2,-1])
    n2_norm = n2/np.linalg.norm(n2)
    cos = np.dot(n1_norm, n2_norm)
    sin = np.sqrt(1-cos**2)

    # axis of rotation
    u = np.cross(n1_norm,n2_norm)
    u = u/np.linalg.norm(u)
    R = np.array([[cos+u[0]**2*(1-cos),          u[0]*u[1]*(1-cos)-u[2]*sin, u[0]*u[2]*(1-cos)+u[1]*sin],
                  [u[1]*u[0]*(1-cos)+u[2]*sin,  cos+u[1]**2*(1-cos),        u[1]*u[2]*(1-cos)-u[0]*sin],
                  [u[2]*u[0]*(1-cos)-u[1]*sin,  u[2]*u[1]*(1-cos)+u[0]*sin, cos+u[2]**2*(1-cos)]])
    return R 


def affine_transform_points(R, moving_pts, fixed_pts=None, moving_plane=(0,0,0), fixed_plane=(0,0,0), downsample_factor=None, plot=False):
    '''
    Performs affine transform on moving_pts using the calcualted affine matrix R
    fixed_plane is a tuple with the parametrized plane for the fixed points (A*X + B*Y + C = Z)
    '''
    
    if plot:
        moving_pts_og = moving_pts.copy() 

    # In case we are using a downsampled affine transform, and the scaling factor is not even, then we can't directly
    # apply the same affine transform to our points 
    if downsample_factor is not None:
        for i in range(3):
            moving_pts[:,i] = moving_pts[:,i] / float(downsample_factor[i])
    # moving_pts[:,2] = moving_pts[:,2] - moving_plane[2] 

    moving_pts_transformed = np.matmul(R, moving_pts.transpose()).transpose()
    moving_pts_transformed[:,2] += fixed_plane[2] - moving_plane[2]

    if downsample_factor is not None:
        for i in range(3):
            moving_pts_transformed[:,i] = moving_pts_transformed[:,i] * float(downsample_factor[i])

    moving_pts_transformed = moving_pts_transformed.astype('int')

    if plot:
        # Also include the plotting of the planes 
        X,Y = np.meshgrid(moving_pts[:,0].max()*np.linspace(0.0, 1.0, 100), 
                          moving_pts[:,1].max()*np.linspace(0.0, 1.0, 100))
        Z = moving_plane[0]*X + moving_plane[1]*Y #+ Cbot[2]

        fixed_pts_og = fixed_pts.copy()
        if downsample_factor is not None:
            for i in range(3):
                fixed_pts[:,i] = fixed_pts[:,i] / float(downsample_factor[i])

        X_top,Y_top = np.meshgrid(fixed_pts[:,0].max()*np.linspace(0.0, 1.0, 100), 
                                 fixed_pts[:,1].max()*np.linspace(0.0, 1.0, 100))
        Z_top = fixed_plane[0]*X_top + fixed_plane[1]*Y_top + fixed_plane[2]

        X_new = R[0,0]*X + R[0,1]*Y + R[0,2]*Z
        Y_new = R[1,0]*X + R[1,1]*Y + R[1,2]*Z
        Z_new = R[2,0]*X + R[2,1]*Y + R[2,2]*Z + fixed_plane[2]

        if downsample_factor is not None:
            X_new *= float(downsample_factor[0])
            Y_new *= float(downsample_factor[1])
            Z_new *= float(downsample_factor[2])

            X_top *= float(downsample_factor[0])
            Y_top *= float(downsample_factor[1])
            Z_top *= float(downsample_factor[2])


        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(moving_pts_transformed[:,0], moving_pts_transformed[:,1], moving_pts_transformed[:,2], c='r', s=50)
        ax.plot_surface(X_new, Y_new, Z_new, rstride=1, cstride=1, alpha=0.2)
        ax.scatter(moving_pts_og[:,0], moving_pts_og[:,1], moving_pts_og[:,2], c='m', s=50)
        ax.plot_surface(X_top,Y_top,Z_top,rstride=1,cstride=1,alpha=0.2)
        ax.scatter(fixed_pts_og[:,0], fixed_pts_og[:,1], fixed_pts_og[:,2], c='b', s=50)
        ax.plot_surface(X,Y,Z+moving_plane[2],rstride=1,cstride=1,alpha=0.2)

        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('equal')
        ax.axis('tight')
        # ax.legend(['Rotated bottom','bottom','top'])
        plt.show()

    return moving_pts_transformed 

