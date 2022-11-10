import numpy as np 
import cupy as cp 
import torch 
import torch.nn.functional as F 
from ..utils import accumarray, get_chunk_coords, numpy_to_json
import multiprocessing as mp 
from tqdm import tqdm 
from functools import partial 
import time 
# from phathom import registration as reg 
from .rigid import rigid_transform, get_inverse_rigid_transform
from .gpu_transform import register 
import zarr 
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import LinearNDInterpolator as lni 

# Module with tools for image registration and deformation - GPU support 

# the temporary zarr path in which we will store intermediate results 
tmp_path = 'temp/temp_transform.zarr'

def point_registration(x_moving, x_fixed, sizeI, **opts):
    '''
    Find appopriate 2D or 3D B-spline grid for transforming a set of moving points to corresponding
    points (fixed).
    
    Inputs:
    sizeI - tuple, size of the image 
    x_moving - N*3 array, coordinates to be warped 
    x_fixed - N*3, corresponding coordinates to the moving points 
    
    **opts:
    max_iter - int, the maximum number of iterations to refine the grid (default: computed)
    threshold_error - float, the min. threhsold error that when achieved stops the refinements (default: None)
    spacing - float, the coarsest spacing to initially perform B splines (default: 2^max_iter)
    use_cupy - bool, whether to use GPU for computations. (default: False) 
    
    Outputs:
    grid - b-spline control points 
    spacing - uniform b spline knot spacing 
    x_reg - the points in x_moving transformed by the fitted B-spline grid 
    '''
    
    if 'max_ref' in opts:
        max_iter = opts['max_ref']
    else:
        max_iter = np.min(np.floor(np.log2(np.asarray(sizeI)/2)))   
    if 'threshold_error' in opts:
        thresh = opts['threshold_error']
    else:
        thresh = None 
    if 'spacing' in opts:
        if opts['spacing'] < 2**max_iter:
            print('Initial spacing is too fine for number of iterations specified. \
                   Defaulting to 2^max_iter initial spacing.')
            spacing = [2**max_iter for _ in range(3)]
        else:
            spacing = [opts['spacing'] for _ in range(3)] 
    else:
        spacing = [2**max_iter for _ in range(3)]
    
    if 'use_cupy' in opts:
        use_cupy = opts['use_cupy']
    else:
        use_cupy = False 
    if 'rigid_M' in opts:
        M = opts['rigid_M'] 
    else:
        M = None
    
    
    start = time.time() 
    o_ref = _make_init_grid(spacing, sizeI, M=M, use_cupy=use_cupy) # reference grid 
    print("TIme elapsed for initializing grid: %f"%(time.time()-start))
    # if use_cupy:
        # np = cp
        # o_ref = cp.asarray(o_ref)
        # x_moving = cp.asarray(x_moving)
        # x_fixed = cp.asarray(x_fixed)
    # else:
        # np = numpy 
    
    # Calculate difference between the points 
    r = x_fixed - x_moving 
    
    # Initialize the grid update needed to b-spline register the points 
    o_add = np.zeros(o_ref.shape) 
    
    # Loop through all refinement iterations 
    for i in range(max_iter):
        print('Iteration: %d / %d'%(i+1, max_iter))
        print('Grid size:',o_ref.shape) 
        
        # Make a b spline grid which minimizes the difference between the corresponding points 
        start = time.time()
        o_add = _bspline_grid_fitting(o_add, spacing, r, x_moving)
        print("Time elapsed for fitting the grid: %f"%(time.time()-start))
        
        # Warp the points 
        start = time.time()
        x_reg = bspline_transform_points_3d(o_ref+o_add, spacing, x_moving) 
        print("Time elapsed for transforming points: %f"%(time.time()-start))

        # Calculate the remaining difference between the points 
        r = x_fixed - x_reg 
        if thresh is not None:
            err = np.sqrt(np.sum(r**2,axis=1))
            mean_err = np.mean(err)
            print('Mean distance:', mean_err)
            if mean_err <= thresh:
                break 
                
        if i < max_iter-1:
            start = time.time()
            o_add, _ = _refine_grid(o_add, spacing, sizeI)
            print("Time elapsed for refining grid: %f"%(time.time()-start))
            start = time.time()
            o_ref, spacing = _refine_grid(o_ref, spacing, sizeI) 
            print("Time elapsed for refining grid: %f"%(time.time()-start))
            
    o_trans = o_ref + o_add 
    
    return o_trans, spacing, x_reg 

    
def _make_init_grid(spacing, sizeI, M=None, use_cupy=False):
    '''
    Create a uniform b spline control grid 
    
    Inputs:
    spacing - 3-tuple, spacing of the b spline grid 
    sizeI - tuple, size of the image 
    M - the inverse transformation from rigid registration (default: None)
    use_cupy - bool (default: False) 
    
    Outputs:
    grid - uniform b spline control point grid 
    '''
    # if use_cupy:
        # np = cp
    # else:
        # np = numpy 
    
    if len(spacing) == 3:
        dx, dy, dz = spacing 
        Ix, Iy, Iz = sizeI
        
        # Calculate the grid coordinates 
        x = np.arange(-dx,Ix+2*dx+1,dx)
        y = np.arange(-dy,Iy+2*dy+1,dy)
        z = np.arange(-dz,Iz+2*dz+1,dz) 
        X,Y,Z = np.meshgrid(x,y,z,indexing='ij') 
        grid = np.ones(X.shape+(3,))
        grid[:,:,:,0] = X
        grid[:,:,:,1] = Y
        grid[:,:,:,2] = Z 
        
        # If we specify an already rigid registration 
        if M is not None:
            # Get center of image 
            # mean = tuple([sizeI[i]//2 for i in range(3)])

            # Make center of the image coordinate 0,0 
            xd = grid[:,:,:,0].copy()# - mean[0]
            yd = grid[:,:,:,1].copy()# - mean[1]
            zd = grid[:,:,:,2].copy()# - mean[2] 
            
            # Calculate the rigid transformed coordinates 
            grid[:,:,:,0] = M[0,0]*xd + M[0,1]*yd + M[0,2]*zd + M[0,3] # + mean[0]
            grid[:,:,:,1] = M[1,0]*xd + M[1,1]*yd + M[1,2]*zd + M[1,3] # + mean[1]
            grid[:,:,:,2] = M[2,0]*xd + M[2,1]*yd + M[2,2]*zd + M[2,3] # + mean[2]

    elif len(spacing) == 2:
        dx, dy = spacing 
        Ix, Iy = sizeI
        
        # Calculate the grid coordinates 
        x = np.arange(-dx,Ix+2*dx+1,dx)
        y = np.arange(-dy,Iy+2*dy+1,dy)
        X,Y= np.meshgrid(x,y,indexing='ij') 
        grid = np.ones(X.shape+(2,))
        grid[:,:,0] = X
        grid[:,:,1] = Y
        
        # If we specify an already rigid registration 
        if M is not None:
            # Get center of image 
            # mean = tuple([sizeI[i]//2 for i in range(2)])
            
            # Make center of the image coordinate 0,0 
            xd = grid[:,:,0].copy() # - mean[0]
            yd = grid[:,:,1].copy() # - mean[1]
            
            # Calculate the rigid transformed coordinates 
            grid[:,:,0] = M[0,0]*xd + M[0,1]*yd + M[0,2]
            grid[:,:,1] = M[1,0]*xd + M[1,1]*yd + M[1,2]
            
    return grid 
    



#### BSpline grid ftting 
    
def _bspline_grid_fitting(Og, spacing, D, X):
    '''
    Currently only support for 3D grid fitting.
    '''
    _EPSILON = 1e-16
    
    # np = cp.get_array_module(Og) 
    
    if D.shape[1] == 3:
        # Calculate which is the closest point on the lattice to the top-left 
        # corner and find ratios of influence between lattice points 
        gx = np.floor(X[:,0]/spacing[0])
        gy = np.floor(X[:,1]/spacing[1])
        gz = np.floor(X[:,2]/spacing[2])
        
        # Calcualte the b-spline coordinate within the b-spline cell [0,1]
        ax = (X[:,0]-gx*spacing[0])/spacing[0]
        ay = (X[:,1]-gy*spacing[1])/spacing[1]
        az = (X[:,2]-gz*spacing[2])/spacing[2]
        
        gx += 2; gy += 2; gz += 2
        
        if ((ax<0).sum()>0) or ((ax>1).sum()>0) or ((ay<0).sum()>0) or ((ay>1).sum()>0) or \
        ((az<0).sum()>0) or ((az>1).sum()>0):
            raise RuntimeError('Grid error')
        W = _bspline_coefficients_3d(ax,ay,az)
        
        # Make indices of all neighborhood knots to every point 
        ix,iy,iz = np.meshgrid(np.arange(-2,2),np.arange(-2,2),np.arange(-2,2),indexing='ij')
        ix = ix.ravel('F')
        iy = iy.ravel('F')
        iz = iz.ravel('F')
        
        indexx = (np.tile(gx, [64,1]).transpose() + np.tile(ix, [X.shape[0], 1])).ravel('F') 
        indexy = (np.tile(gy, [64,1]).transpose() + np.tile(iy, [X.shape[0], 1])).ravel('F')
        indexz = (np.tile(gz, [64,1]).transpose() + np.tile(iz, [X.shape[0], 1])).ravel('F')
        
        # Limit to boundaries grid 
        indexx = np.minimum(np.maximum(0,indexx), Og.shape[0]-1)
        indexy = np.minimum(np.maximum(0,indexy), Og.shape[1]-1)
        indexz = np.minimum(np.maximum(0,indexz), Og.shape[2]-1)
        index = np.transpose(np.vstack((indexx,indexy,indexz)))
        
        # Update a numerator and a denominator for each knot. 
        W2 = W**2
        S = np.sum(W2, axis=1)
        WT = W2*W
        WNx = WT*np.tile(D[:,0]/S, [64,1]).transpose()
        WNy = WT*np.tile(D[:,1]/S, [64,1]).transpose()
        WNz = WT*np.tile(D[:,2]/S, [64,1]).transpose()

        np.save('temp/wnx.npy',WNx)
        np.save('temp/wny.npy',WNy)
        np.save('temp/wnz.npy',WNz)
        np.save('temp/index.npy',index)
        
        #start = time.time()
        numx = accumarray(index.astype('int'), WNx.ravel('F'), size=np.asarray(Og.shape[:-1]))
        numy = accumarray(index.astype('int'), WNy.ravel('F'), size=np.asarray(Og.shape[:-1]))
        numz = accumarray(index.astype('int'), WNz.ravel('F'), size=np.asarray(Og.shape[:-1]))
        dnum = accumarray(index.astype('int'), W2.ravel('F'), size=np.asarray(Og.shape[:-1]))
        #print("accumarray took %f seconds"%(time.time()-start))

        # Calculate actual values of knots from the numerator and denominator 
        ux = numx / (dnum + _EPSILON)
        uy = numy / (dnum + _EPSILON)
        uz = numz / (dnum + _EPSILON)
        
        # Update the B-spline transformation grid 
        O_trans = np.concatenate((np.expand_dims(ux+Og[:,:,:,0],3),
                                  np.expand_dims(uy+Og[:,:,:,1],3),
                                  np.expand_dims(uz+Og[:,:,:,2],3)), axis=3)
        
    return O_trans 

def _bspline_coefficients_1d(u):
    # Cubic B Spline coefficients 
    
    # np = cp.get_array_module(u)
    
    W = np.zeros((u.shape[0],4))
    W[:,0] = (1-u)**3/6
    W[:,1] = (3*u**3 - 6*u**2 + 4)/6 
    W[:,2] = (-3*u**3 + 3*u**2 + 3*u + 1)/6
    W[:,3] = u**3/6 
    return W 

def _bspline_coefficients_2d(u,v):
    # np = cp.get_array_module(u)
    
    Bu = _bspline_coefficients_1d(u)
    Bv = _bspline_coefficients_1d(v)
    
    # Calculate influences of all neighborhood b-spline knots 
    W = np.transpose(np.vstack( (Bu[:,0]*Bv[:,0], Bu[:,1]*Bv[:,0], Bu[:,2]*Bv[:,0], Bu[:,3]*Bv[:,0],
                                Bu[:,0]*Bv[:,1], Bu[:,1]*Bv[:,1], Bu[:,2]*Bv[:,1], Bu[:,3]*Bv[:,1],
                                Bu[:,0]*Bv[:,2], Bu[:,1]*Bv[:,2], Bu[:,2]*Bv[:,2], Bu[:,3]*Bv[:,2],
                                Bu[:,0]*Bv[:,3], Bu[:,1]*Bv[:,3], Bu[:,2]*Bv[:,3], Bu[:,3]*Bv[:,3]) ))
    return W
    
def _bspline_coefficients_3d(u,v,w):
    # np = cp.get_array_module(u) 

    Bu = _bspline_coefficients_1d(u)
    Bv = _bspline_coefficients_1d(v)
    Bw = _bspline_coefficients_1d(w) 
    
    # Calculate influences of all neighborhood b-spline knots 
    W = np.transpose(np.vstack( (Bu[:,0]*Bv[:,0]*Bw[:,0], Bu[:,1]*Bv[:,0]*Bw[:,0], Bu[:,2]*Bv[:,0]*Bw[:,0], Bu[:,3]*Bv[:,0]*Bw[:,0],
                                 Bu[:,0]*Bv[:,1]*Bw[:,0], Bu[:,1]*Bv[:,1]*Bw[:,0], Bu[:,2]*Bv[:,1]*Bw[:,0], Bu[:,3]*Bv[:,1]*Bw[:,0],
                                 Bu[:,0]*Bv[:,2]*Bw[:,0], Bu[:,1]*Bv[:,2]*Bw[:,0], Bu[:,2]*Bv[:,2]*Bw[:,0], Bu[:,3]*Bv[:,2]*Bw[:,0],
                                 Bu[:,0]*Bv[:,3]*Bw[:,0], Bu[:,1]*Bv[:,3]*Bw[:,0], Bu[:,2]*Bv[:,3]*Bw[:,0], Bu[:,3]*Bv[:,3]*Bw[:,0],
                                 Bu[:,0]*Bv[:,0]*Bw[:,1], Bu[:,1]*Bv[:,0]*Bw[:,1], Bu[:,2]*Bv[:,0]*Bw[:,1], Bu[:,3]*Bv[:,0]*Bw[:,1],
                                 Bu[:,0]*Bv[:,1]*Bw[:,1], Bu[:,1]*Bv[:,1]*Bw[:,1], Bu[:,2]*Bv[:,1]*Bw[:,1], Bu[:,3]*Bv[:,1]*Bw[:,1],
                                 Bu[:,0]*Bv[:,2]*Bw[:,1], Bu[:,1]*Bv[:,2]*Bw[:,1], Bu[:,2]*Bv[:,2]*Bw[:,1], Bu[:,3]*Bv[:,2]*Bw[:,1],
                                 Bu[:,0]*Bv[:,3]*Bw[:,1], Bu[:,1]*Bv[:,3]*Bw[:,1], Bu[:,2]*Bv[:,3]*Bw[:,1], Bu[:,3]*Bv[:,3]*Bw[:,1],
                                 Bu[:,0]*Bv[:,0]*Bw[:,2], Bu[:,1]*Bv[:,0]*Bw[:,2], Bu[:,2]*Bv[:,0]*Bw[:,2], Bu[:,3]*Bv[:,0]*Bw[:,2],
                                 Bu[:,0]*Bv[:,1]*Bw[:,2], Bu[:,1]*Bv[:,1]*Bw[:,2], Bu[:,2]*Bv[:,1]*Bw[:,2], Bu[:,3]*Bv[:,1]*Bw[:,2],
                                 Bu[:,0]*Bv[:,2]*Bw[:,2], Bu[:,1]*Bv[:,2]*Bw[:,2], Bu[:,2]*Bv[:,2]*Bw[:,2], Bu[:,3]*Bv[:,2]*Bw[:,2],
                                 Bu[:,0]*Bv[:,3]*Bw[:,2], Bu[:,1]*Bv[:,3]*Bw[:,2], Bu[:,2]*Bv[:,3]*Bw[:,2], Bu[:,3]*Bv[:,3]*Bw[:,2],
                                 Bu[:,0]*Bv[:,0]*Bw[:,3], Bu[:,1]*Bv[:,0]*Bw[:,3], Bu[:,2]*Bv[:,0]*Bw[:,3], Bu[:,3]*Bv[:,0]*Bw[:,3],
                                 Bu[:,0]*Bv[:,1]*Bw[:,3], Bu[:,1]*Bv[:,1]*Bw[:,3], Bu[:,2]*Bv[:,1]*Bw[:,3], Bu[:,3]*Bv[:,1]*Bw[:,3],
                                 Bu[:,0]*Bv[:,2]*Bw[:,3], Bu[:,1]*Bv[:,2]*Bw[:,3], Bu[:,2]*Bv[:,2]*Bw[:,3], Bu[:,3]*Bv[:,2]*Bw[:,3],
                                 Bu[:,0]*Bv[:,3]*Bw[:,3], Bu[:,1]*Bv[:,3]*Bw[:,3], Bu[:,2]*Bv[:,3]*Bw[:,3], Bu[:,3]*Bv[:,3]*Bw[:,3]) ))
    return W    

def bspline_transform_points_3d(o_trans, spacing, X): 
    '''
    Computes the coordinates of each point in X after the transformation.
    
    Inputs:
    o_trans - 4D array containing the transformation grid of control points 
    spacing - 3-tuple or list that contains the spacing between control points 
    X - N*3 array containing the coordinates of the fixed points to be transformed 
    
    Output:
    Tlocal - N*3 array that contains the points transformation 
    '''
    
    # np = cp.get_array_module(o_trans)
    
    x2 = X[:,0]; y2 = X[:,1]; z2 = X[:,2]
    
    m,l,k = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing='ij')
    m = m.ravel('F')
    l = l.ravel('F')
    k = k.ravel('F')
    ixs = np.floor(x2/spacing[0])
    iys = np.floor(y2/spacing[1])
    izs = np.floor(z2/spacing[2])
    ix = (np.tile(ixs,[64,1]).transpose() + np.tile(m, [len(x2),1])).ravel('F')
    iy = (np.tile(iys,[64,1]).transpose() + np.tile(l, [len(y2),1])).ravel('F')
    iz = (np.tile(izs,[64,1]).transpose() + np.tile(k, [len(z2),1])).ravel('F') 
    
    s = o_trans.shape 
    
    # Points outside the b spline grid are set to the upper corner 
    check_bound = (ix<0) + (ix>(s[0]-1)) + (iy<0) + (iy>(s[1]-1)) + (iz<0) + (iz>(s[2]-1))
    ix[check_bound] = 1
    iy[check_bound] = 1
    iz[check_bound] = 1
 
    Cx = o_trans[(ix.astype('int'), iy.astype('int'), iz.astype('int'), 0)] * ~check_bound
    Cy = o_trans[(ix.astype('int'), iy.astype('int'), iz.astype('int'), 1)] * ~check_bound
    Cz = o_trans[(ix.astype('int'), iy.astype('int'), iz.astype('int'), 2)] * ~check_bound
    
    Cx = Cx.reshape([len(x2),64],order='F')
    Cy = Cy.reshape([len(x2),64],order='F')
    Cz = Cz.reshape([len(x2),64],order='F')
    
    # Calculate the B spline interpolation constants u,v in the center cell 
    u = (x2-ixs*spacing[0])/spacing[0]
    v = (y2-iys*spacing[1])/spacing[1]
    w = (z2-izs*spacing[2])/spacing[2]
    
    # Get the b spline coefficients in a matrix W, which contains the influence of all knots on the points 
    # on the points in (x2,y2) 
    W = _bspline_coefficients_3d(u,v,w)
    
    # Calculate the transformation of the points in (x2,y2) by the b spline grid 
    Tlocal = np.vstack((np.sum(W*Cx,axis=1),np.sum(W*Cy,axis=1),np.sum(W*Cz,axis=1))).transpose()
    
    return Tlocal 
    
def _refine_grid(o_trans, spacing, sizeI): 
    '''
    Refine image transformation grid of 1D b-splines by using splitting matrix 
    Msplit = (1/8) * [4 4 0 0
                      1 6 1 0
                      0 4 4 0
                      0 1 6 1
                      0 0 4 4]
                      
    Outputs: 
    o_new - new grid 
    spacing - new spacing 
    '''
    # np = cp.get_array_module(o_trans)
    
    # Spacing is halved 
    spacing = [spacing[i]/2 for i in range(len(spacing))]
    
    # 3D 
    # Refine in the x direction 
    o_newA = np.zeros(((o_trans.shape[0]*2-3,)+o_trans.shape[1:]))
    [I,J,K,H] = np.meshgrid(np.arange(o_trans.shape[0]-3),np.arange(o_trans.shape[1]),np.arange(o_trans.shape[2]),np.arange(3),indexing='ij')
    # [J,I,K,H] = np.meshgrid(np.arange(o_trans.shape[1]),np.arange(o_trans.shape[0]-3),np.arange(o_trans.shape[2]),np.arange(3))
    I = I.ravel('F'); J = J.ravel('F'); K = K.ravel('F'); H = H.ravel('F')
    p0 = o_trans[(I,J,K,H)]
    p1 = o_trans[(I+1,J,K,H)]
    p2 = o_trans[(I+2,J,K,H)]
    p3 = o_trans[(I+3,J,K,H)]
    pnew = _split_knots(p0,p1,p2,p3)
    
    o_newA[(2*I,J,K,H)] = pnew[:,0]
    o_newA[(2*I+1,J,K,H)] = pnew[:,1]
    o_newA[(2*I+2,J,K,H)] = pnew[:,2]
    o_newA[(2*I+3,J,K,H)] = pnew[:,3]
    o_newA[(2*I+4,J,K,H)] = pnew[:,4]
    
    # Refine in teh y direction 
    o_newB = np.zeros(((o_newA.shape[0],)+(o_newA.shape[1]*2-3,)+o_newA.shape[2:]))
    [I,J,K,H] = np.meshgrid(np.arange(o_newA.shape[0]),np.arange(o_newA.shape[1]-3),np.arange(o_newA.shape[2]),np.arange(3),indexing='ij')
    # [J,I,K,H] = np.meshgrid(np.arange(o_newA.shape[1]-3),np.arange(o_newA.shape[0]),np.arange(o_newA.shape[2]),np.arange(3))
    I = I.ravel('F'); J = J.ravel('F'); K = K.ravel('F'); H = H.ravel('F')
    p0 = o_newA[(I,J,K,H)]
    p1 = o_newA[(I,J+1,K,H)]
    p2 = o_newA[(I,J+2,K,H)]
    p3 = o_newA[(I,J+3,K,H)]
    pnew = _split_knots(p0,p1,p2,p3)
    
    o_newB[(I,2*J,K,H)] = pnew[:,0]
    o_newB[(I,2*J+1,K,H)] = pnew[:,1]
    o_newB[(I,2*J+2,K,H)] = pnew[:,2]
    o_newB[(I,2*J+3,K,H)] = pnew[:,3]
    o_newB[(I,2*J+4,K,H)] = pnew[:,4]
    
    # Refine in the z direction 
    o_newC = np.zeros((o_newB.shape[:2]+(o_newB.shape[2]*2-3,3)))
    [I,J,K,H] = np.meshgrid(np.arange(o_newB.shape[0]),np.arange(o_newB.shape[1]),np.arange(o_newB.shape[2]-3),np.arange(3), indexing='ij')
    # [J,I,K,H] = np.meshgrid(np.arange(o_newB.shape[1]),np.arange(o_newB.shape[0]),np.arange(o_newB.shape[2]-3),np.arange(3))
    I = I.ravel('F'); J = J.ravel('F'); K = K.ravel('F'); H = H.ravel('F')
    p0 = o_newB[(I,J,K,H)]
    p1 = o_newB[(I,J,K+1,H)]
    p2 = o_newB[(I,J,K+2,H)]
    p3 = o_newB[(I,J,K+3,H)]
    pnew = _split_knots(p0,p1,p2,p3)
    
    o_newC[(I,J,K*2,H)] = pnew[:,0]
    o_newC[(I,J,K*2+1,H)] = pnew[:,1]
    o_newC[(I,J,K*2+2,H)] = pnew[:,2]
    o_newC[(I,J,K*2+3,H)] = pnew[:,3]
    o_newC[(I,J,K*2+4,H)] = pnew[:,4]
    
    # Crop to the correct size 
    dx, dy, dz = spacing 
    X,_,_ = np.meshgrid(np.arange(-dx,sizeI[0]+2*dx+1,dx), 
                        np.arange(-dy,sizeI[1]+2*dy+1,dy),
                        np.arange(-dz,sizeI[2]+2*dz+1,dz), indexing='ij')               
                    
    return o_newC[:X.shape[0],:X.shape[1],:X.shape[2],:], spacing
    

def _split_knots(p0,p1,p2,p3):
    # np = cp.get_array_module(p0)
    pn0 = 1/2*(p0+p1) 
    pn1 = 1/8*(p0+6*p1+p2)
    pn2 = 1/2*(p1+p2)
    pn3 = 1/8*(p1+6*p2+p3)
    pn4 = 1/2*(p2+p3) 
    return np.transpose( np.vstack((pn0,pn1,pn2,pn3,pn4)) ) 
    



#### Parallel b spline transformation. 
# need to parallelize _make_init_grid, bspline_grid_fitting, bspline_points_transform_3d, _refine_grid

def point_registration_zarr(x_moving, x_fixed, sizeI, grid_zarr_path, **opts):
    '''
    Same as point_registration, but parallelized for zarrs. 
    
    grid_zarr_path - str, the sink path for the final grid 

    Opts now includes:
    num_workers - int, number of parallel workers (default: 8)
    chunks - (int,int,int), tuple of chunk sizes (default: (128,128,128))
    '''

    if 'max_ref' in opts:
        max_iter = opts['max_ref']
    else:
        max_iter = np.min(np.floor(np.log2(np.asarray(sizeI)/2)))   
    if 'threshold_error' in opts:
        thresh = opts['threshold_error']
    else:
        thresh = None 
    if 'spacing' in opts:
        if opts['spacing'] < 2**max_iter:
            print('Initial spacing is too fine for number of iterations specified. \
                   Defaulting to 2^max_iter initial spacing.')
            spacing = [2**max_iter for _ in range(3)]
        else:
            spacing = [opts['spacing'] for _ in range(3)] 
    else:
        spacing = [2**max_iter for _ in range(3)]
    
    if 'use_cupy' in opts:
        use_cupy = opts['use_cupy']
    else:
        use_cupy = False 
    if 'num_workers' in opts:
        num_workers = opts['num_workers']
    else:
        num_workers = 8
    if 'chunks' in opts:
        chunks = opts['chunks']
    else:
        chunks = (128,128,128)


    print("Initializing grid...")
    o_ref = _make_init_grid_zarr(grid_zarr_path, chunks, spacing, sizeI, num_workers=num_workers) # reference grid 
    print("Done initializing grid.")

    # Calculate difference between the points 
    r = x_fixed - x_moving 
    
    # Initialize the grid update needed to b-spline register the points 

    o_add = zarr.zeros(shape=o_ref.shape, chunks=o_ref.chunks, 
                        store=zarr.DirectoryStore(tmp_path), overwrite=True)
    
    # Loop through all refinement iterations 
    for i in range(max_iter):
        print('Iteration: %d / %d'%(i+1, max_iter))
        print('Grid size:',o_ref.shape) 
        
        # Make a b spline grid which minimizes the difference between the corresponding points 
        o_add = _bspline_grid_fitting_zarr(o_add, spacing, r, x_moving)
        
        # Warp the points 
        x_reg = bspline_transform_points_3d_zarr(o_ref, o_add, spacing, x_moving) 
        
        # Calculate the remaining difference between the points 
        r = x_fixed - x_reg 
        if thresh is not None:
            err = np.sqrt(np.sum(r**2,axis=1))
            mean_err = np.mean(err)
            print('Mean distance:', mean_err)
            if mean_err <= thresh:
                break 
                
        if i < max_iter-1:
            o_add, _ = _refine_grid_zarr(o_add, spacing, sizeI)
            o_ref, spacing = _refine_grid_zarr(o_ref, spacing, sizeI) 
            
    o_trans = o_ref + o_add 
    
    #return o_trans, spacing, x_reg 

    pass 

def _make_init_grid_zarr(sink_zarr_path, chunks, spacing, sizeI, M=None, num_workers=8):
    '''
    Create a uniform b spline control grid 
    
    Inputs:
    sink_zarr_path - str, the path to which we write the grid.
    chunks - tuple, chunk-size of the grid 
    spacing - 3-tuple, spacing of the b spline grid 
    sizeI - tuple, size of the image 
    M - the inverse transformation from rigid registration (default: None)
    num_workers - int, the number of parallel workers to use (default: 8) 
    
    Outputs:
    grid - uniform b spline control point grid 
    '''
   

    if len(spacing) == 3:
        dx, dy, dz = spacing 
        Ix, Iy, Iz = sizeI
        
        # Calculate the grid coordinates 
        x = np.arange(-dx,Ix+2*dx+1,dx)
        y = np.arange(-dy,Iy+2*dy+1,dy)
        z = np.arange(-dz,Iz+2*dz+1,dz) 


        grid_zarr = zarr.open(sink_zarr_path, mode='w', shape=(len(x),len(y),len(z),3), chunks=chunks, dtype='float') 
        coord_ranges = get_chunk_coords(grid_zarr.shape, grid_zarr.chunks)
        p = mp.Pool(num_workers)
        f = partial(_make_init_grid_zarr_serial, grid_zarr, spacing)
        list(tqdm(p.imap(f, coord_ranges), total=len(coord_ranges)))
        
        
        # If we specify an already rigid registration 
        if M is not None:
            # Get center of image 
            mean = tuple([sizeI[i]//2 for i in range(3)])
            # Make center of the image coordinate 0,0 
            xd = grid[:,:,:,0] - mean[0]
            yd = grid[:,:,:,1] - mean[1]
            zd = grid[:,:,:,2] - mean[2] 
            
            # Calculate the rigid transformed coordinates 
            grid[:,:,:,0] = mean[0] + M[0,0]*xd + M[0,1]*yd + M[0,2]*zd + M[0,3]
            grid[:,:,:,1] = mean[1] + M[1,0]*xd + M[1,1]*yd + M[1,2]*zd + M[1,3]
            grid[:,:,:,2] = mean[2] + M[2,0]*xd + M[2,1]*yd + M[2,2]*zd + M[2,3]

        return grid 
    

def _make_init_grid_zarr_serial(grid_zarr, x, y, z, coord):
    '''
    Kernel for parallelized grid initialization. Assigns the appropriate points to 
    '''

    xr, yr, zr = coord # coordinate ranges at which to initialize the grid 
    X,Y,Z = np.meshgrid(x[xr[0]:xr[1]],y[yr[0]:yr[1]],z[zr[0]:zr[1]],indexing='ij') 

    grid_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1],0] = X
    grid_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1],1] = Y
    grid_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1],2] = Z


def _bspline_grid_fitting_zarr(Og, spacing, D, X):
    '''
    Og - 4D array containing how much to add to each coordinate at each grid point (in zarr form)  
    spacing - tuple of the spacings 
    D - N*3 array of the distances between moving and fixed points 
    X - N*3 array of the moving points 
    '''

    _EPSILON = 1e-16
    
    if D.shape[1] == 3:
        # Calculate which is the closest point on the lattice to the top-left 
        # corner and find ratios of influence between lattice points 
        gx = np.floor(X[:,0]/spacing[0])
        gy = np.floor(X[:,1]/spacing[1])
        gz = np.floor(X[:,2]/spacing[2])
        
        # Calcualte the b-spline coordinate within the b-spline cell [0,1]
        ax = (X[:,0]-gx*spacing[0])/spacing[0]
        ay = (X[:,1]-gy*spacing[1])/spacing[1]
        az = (X[:,2]-gz*spacing[2])/spacing[2]
        
        gx += 2; gy += 2; gz += 2
        
        if ((ax<0).sum()>0) or ((ax>1).sum()>0) or ((ay<0).sum()>0) or ((ay>1).sum()>0) or \
        ((az<0).sum()>0) or ((az>1).sum()>0):
            raise RuntimeError('Grid error')
        W = _bspline_coefficients_3d(ax,ay,az)
        
        # Make indices of all neighborhood knots to every point 
        ix,iy,iz = np.meshgrid(np.arange(-2,2),np.arange(-2,2),np.arange(-2,2),indexing='ij')
        ix = ix.ravel('F')
        iy = iy.ravel('F')
        iz = iz.ravel('F')
        
        indexx = (np.tile(gx, [64,1]).transpose() + np.tile(ix, [X.shape[0], 1])).ravel('F') 
        indexy = (np.tile(gy, [64,1]).transpose() + np.tile(iy, [X.shape[0], 1])).ravel('F')
        indexz = (np.tile(gz, [64,1]).transpose() + np.tile(iz, [X.shape[0], 1])).ravel('F')
        
        # Limit to boundaries grid 
        indexx = np.minimum(np.maximum(0,indexx), Og.shape[0]-1)
        indexy = np.minimum(np.maximum(0,indexy), Og.shape[1]-1)
        indexz = np.minimum(np.maximum(0,indexz), Og.shape[2]-1)
        index = np.transpose(np.vstack((indexx,indexy,indexz)))
        
        # Update a numerator and a denominator for each knot. 
        W2 = W**2
        S = np.sum(W2, axis=1)
        WT = W2*W
        WNx = WT*np.tile(D[:,0]/S, [64,1]).transpose()
        WNy = WT*np.tile(D[:,1]/S, [64,1]).transpose()
        WNz = WT*np.tile(D[:,2]/S, [64,1]).transpose()

        ## TODO: Parallelize this part 
        numx = accumarray(index.astype('int'), WNx.ravel('F'), size=np.asarray(Og.shape[:-1]))
        numy = accumarray(index.astype('int'), WNy.ravel('F'), size=np.asarray(Og.shape[:-1]))
        numz = accumarray(index.astype('int'), WNz.ravel('F'), size=np.asarray(Og.shape[:-1]))
        dnum = accumarray(index.astype('int'), W2.ravel('F'), size=np.asarray(Og.shape[:-1]))
        
        # Calculate actual values of knots from the numerator and denominator 
        ux = numx / (dnum + _EPSILON)
        uy = numy / (dnum + _EPSILON)
        uz = numz / (dnum + _EPSILON)
        
        # Update the B-spline transformation grid 
        ## TODO: write a function to also update this 
        O_trans = np.concatenate((np.expand_dims(ux+Og[:,:,:,0],3),
                                  np.expand_dims(uy+Og[:,:,:,1],3),
                                  np.expand_dims(uz+Og[:,:,:,2],3)), axis=3)
        
    return O_trans 

def _bspline_grid_fitting_zarr_serial(sink_zarr, Og, index, WNx, WNy, WNz, W2, coord):

    _EPSILON = 1e-16
    xr,yr,zr = coord 

    ## do accumarray 
    numx = accumarray(index.astype('int'), WNx.ravel('F'))

    ## Assign 
    sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1],0] = ux + Og[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1],0]
    sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1],1] = uy + Og[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1],1]
    sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1],2] = uz + Og[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1],2]
    pass


# Nonuniform B Splines - 
## Primary goal is still to be able to parallelize and not have to use this. 
 
def _make_init_grid_nonuniform(spacings, slice_nums, sizeI, M=None, use_cupy=False):
    '''
    Create a non-uniform b spline control grid 
    
    Inputs:
    spacings - tuple of 3-tuples, spacings of the b spline grid at each resolution
    slice_nums - tuple of same length as spacings, with the z slices at which the resolutions will be applied.  
    sizeI - tuple, total size of the image 
    M - the inverse transformation from rigid registration (default: None)
    use_cupy - bool (default: False) 
    
    Outputs:
    grid - uniform b spline control point grid 
    ''' 
    
    if len(slice_nums) != len(spacings):
        raise RuntimeError('Number of spacings not equal to number of slices.')
    num_resolutions = len(slice_nums)
    Ix,Iy,Iz = sizeI

    # Initialize the grid 
    total_grid = []
    for i in range(num_resolutions):
        dx,dy,dz = spacings[i]
        if i == 0:
            first_slice = 0
        else:
            first_slice = slice_nums[i-1]
        second_slice = slice_nums[i]
        Iz = second_slice-first_slice

        # Calculate the local grid coordinates 
        x = np.arange(-dx,Ix+2*dx+1,dx)
        y = np.arange(-dy,Iy+2*dy+1,dy)
        z = np.arange(-dz,Iz+2*dz+1,dz) 
        X,Y,Z = np.meshgrid(x,y,z,indexing='ij') 
        grid = np.ones(X.shape+(3,))
        grid[:,:,:,0] = X
        grid[:,:,:,1] = Y
        grid[:,:,:,2] = Z 
        total_grid.append(grid)


        # If we specify an already rigid registration 
        if M is not None:
            # Get center of image 
            mean = tuple([sizeI[i]//2 for i in range(3)])
            # Make center of the image coordinate 0,0 
            xd = grid[:,:,:,0] - mean[0]
            yd = grid[:,:,:,1] - mean[1]
            zd = grid[:,:,:,2] - mean[2] 

            # Calculate the rigid transformed coordinates 
            grid[:,:,:,0] = mean[0] + M[0,0]*xd + M[0,1]*yd + M[0,2]*zd + M[0,3]
            grid[:,:,:,1] = mean[1] + M[1,0]*xd + M[1,1]*yd + M[1,2]*zd + M[1,3]
            grid[:,:,:,2] = mean[2] + M[2,0]*xd + M[2,1]*yd + M[2,2]*zd + M[2,3]

    return total_grid


##################### TPS tools #############################

# More functions for transforming
def warp_regular_grid(nb_pts, x, y, z, transform):
    X,Y,Z= np.meshgrid(x,y,z, indexing='ij')
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    values = transform(grid)

    values_x = np.reshape(values[:, 0], nb_pts)
    values_y = np.reshape(values[:, 1], nb_pts)
    values_z = np.reshape(values[:, 2], nb_pts)

    return values_x, values_y, values_z


def TPS_zarr_warp(moving_zarr_path, warped_zarr_path,
                  moving_keypts, fixed_keypts, warped_zarr_size=None,
                  grid_spacing=3*(16,), smooth=2,
                  Rinv=None, binv=None, chunks=3*(128,),
                  nb_workers=1, padding=2,
                  save_grid_values_path=None, use_grid_values_path=None, show_residuals=True, voxel_size=(1,1,1)):
    '''
    Use Thin Plate Splines to warp image.
    '''
    start_total = time.time()
    
    if use_grid_values_path is None:
        # Fit TPS
        moving_keypts[:,0] *= voxel_size[0]; moving_keypts[:,1]*=voxel_size[1]; moving_keypts[:,2]*=voxel_size[2]
        fixed_keypts[:,0] *= voxel_size[0]; fixed_keypts[:,1]*=voxel_size[1]; fixed_keypts[:,2]*=voxel_size[2]
        if Rinv is not None and binv is not None:
            affined_keypts = rigid_transform(Rinv,binv,fixed_keypts) # affine transform
            affine_transform = partial(rigid_transform, Rinv, binv)
        else:
            affined_keypts = fixed_keypts
        

        affined_keypts 
        print('Fitting radial basis function...')
        start = time.time()
        rbf_x, rbf_y, rbf_z = fit_rbf(affined_keypts, 
                                          moving_keypts, 
                                          smooth)
        print('Fitting rbf took %f seconds'%(time.time()-start))

        if Rinv is not None and binv is not None:
            combo_transform = partial(nonrigid_transform, affine_transform=affine_transform,
                                        rbf_z=rbf_x, 
                                        rbf_y=rbf_y, 
                                        rbf_x=rbf_z)
        else:
            combo_transform = partial(rbf_transform,
                                        rbf_z=rbf_x, 
                                        rbf_y=rbf_y, 
                                        rbf_x=rbf_z)

        nonrigid_keypts = combo_transform(fixed_keypts)

        if show_residuals:
            nonrigid_residuals = match_distance(nonrigid_keypts,
                                                    moving_keypts)
            print('Nonrigid ave. distance [pixels]:', nonrigid_residuals.mean())
            
        # Warp a regular grid using exact TPS
        if warped_zarr_size is None:
            z = zarr.open(moving_zarr_path, mode='r')
            warped_zarr_size = z.shape
        if len(grid_spacing)==1:
            grid_spacing = 3*(grid_spacing,)
        nb_pts = tuple([int(warped_zarr_size[i]/grid_spacing[i]) for i in range(3)])

        z = np.linspace(0, warped_zarr_size[2]*voxel_size[2], nb_pts[2])
        y = np.linspace(0, warped_zarr_size[1]*voxel_size[1], nb_pts[1])
        x = np.linspace(0, warped_zarr_size[0]*voxel_size[0], nb_pts[0])

        print("Warping grid...")
        start = time.time()

        grid_values = warp_regular_grid(nb_pts, x, y, z, combo_transform)
        grid_values = np.asarray(grid_values)
        grid_values[0] /= voxel_size[0]; grid_values[1] /= voxel_size[1]; grid_values[2] /= voxel_size[2]
        grid_values = np.expand_dims(np.expand_dims(np.asarray(grid_values), 1),1)
        print("Warping grid took %f seconds"%(time.time()-start))
        #print(grid_values.shape)

        if save_grid_values_path is not None:
            np.save(save_grid_values_path, grid_values)
            print('Saved grid_values at %s'%save_grid_values_path)
    else:
        print('Loading grid values...')
        grid_values = np.load(use_grid_values_path)
    
    # Now actually warp it 


    #############
    move_zarr = zarr.open(moving_zarr_path,mode='r')
    warped_zarr = zarr.open(warped_zarr_path, mode='w',
                             shape=warped_zarr_size, 
                            chunks=chunks,dtype='uint16',overwrite=True)

    print("Warping image...")
    register(move_zarr, move_zarr, warped_zarr, grid_values, chunks, nb_workers, padding)
    print("Time elapsed: %f minutes"%((time.time()-start_total)/60))
    
    return warped_zarr

def nonrigid_transform_points(input_pts, target_pts, transform_pts=None, R=None, b=None, smooth=2):
    '''
    Transform input points using thin plate spline to minimize distance to target_pts. 
    
    Optional: transform the transform_pts (not the input_pts) using the calculated TPS 
    Optional: use an initial affine transformation R,b before nonrigid deformation
    '''

    # Fit TPS
    if R is not None and b is not None:
        affined_keypts = rigid_transform(R,b,input_pts) # affine transform
        affine_transform = partial(rigid_transform, R, b)
    else:
        affined_keypts = input_pts
     
    print('Fitting radial basis function...')
    start = time.time()
    rbf_x, rbf_y, rbf_z = fit_rbf(affined_keypts, 
                                      target_pts, 
                                      smooth)
    print('Fitting rbf took %f seconds'%(time.time()-start))

    if R is not None and b is not None:
        combo_transform = partial(nonrigid_transform, affine_transform=affine_transform,
                                    rbf_z=rbf_x, 
                                    rbf_y=rbf_y, 
                                    rbf_x=rbf_z)
    else:
        combo_transform = partial(rbf_transform,
                                    rbf_z=rbf_x, 
                                    rbf_y=rbf_y, 
                                    rbf_x=rbf_z)
    if transform_pts is None:
        nonrigid_keypts = combo_transform(input_pts)
        nonrigid_residuals = match_distance(nonrigid_keypts,
                                            target_pts)
        print('Nonrigid ave. distance [pixels]:', nonrigid_residuals.mean())
    else:
        nonrigid_keypts = combo_transform(transform_pts)

    return nonrigid_keypts



# outward facing function for warping points
def TPS_transform_points(moving_pts_paths, fixed_pts_paths, pts_masked_paths=None, pts_masked_save_paths=None, static_pts_paths=None, R_path=None, b_path=None):
    '''
    pts_masked_paths - if not None, list of numpy arrays of points that we want to transform. If None, transform moving_pts_paths
    '''

    # fixed and moving for defining transform
    fixed_pts = np.zeros((0,3))
    moving_pts = np.zeros((0,3))
    indices = [0,] # indices where to split the points 
    for idx in range(len(moving_pts_paths)):
        temp_pts = np.load(fixed_pts_paths[idx])
        indices.append(temp_pts.shape[0])
        fixed_pts = np.concatenate((fixed_pts,temp_pts),axis=0)
        moving_pts = np.concatenate((moving_pts,np.load(moving_pts_paths[idx])),axis=0)

    # Load dthe points to be transformed  
    pts_masked = np.zeros((0,3))
    if pts_masked_paths is not None:
        indices = [0,] # indices where to split the points 
        for pts_masked_path in pts_masked_paths:
            temp_pts = np.load(pts_masked_path)
            indices.append(temp_pts.shape[0])
            pts_masked = np.concatenate((pts_masked,temp_pts),axis=0)
    else:
        pts_masked = None 

    # Load the static points on other surface, and the manual anchor points 
    anchor_moving = np.zeros((0,3))
    if static_pts_paths is not None:
        for static_pts_path in static_pts_paths:
            static_pts = np.load(static_pts_path)
            anchor_moving = np.concatenate((anchor_moving, static_pts),axis=0)

    # Load affine transform parameters
    if R_path is not None:
        R = np.load(R_path)
        b = np.load(b_path)
        anchor_fixed = rigid_transform(R,b,anchor_moving)
    else:
        R = None; b = None 
        anchor_fixed = anchor_moving.copy()
    
    # For defining the transform 
    if static_pts_paths is not None:
        moving_pts_total = np.concatenate((moving_pts,anchor_moving),axis=0)
        fixed_pts_total =np.concatenate((fixed_pts,anchor_fixed),axis=0)
    else:
        moving_pts_total = moving_pts; fixed_pts_total = fixed_pts 

    # Warp the desired points 
    new_pts = nonrigid_transform_points(moving_pts_total, fixed_pts_total, transform_pts=pts_masked, R=R, b=b)
    final_pts = [new_pts[np.sum(indices[:idx+1]):np.sum(indices[:idx+2])] for idx in range(len(indices)-1)]
    if pts_masked_save_paths is not None:
        for j,save_path in enumerate(pts_masked_save_paths):
            fin_pts = final_pts[j]
            np.save(save_path, fin_pts)
    return final_pts 



# outward facing function for warping (in notebook)
def TPS_warp(moving_zarr_path, fixed_zarr_path, warped_zarr_path, moving_pts_paths, fixed_pts_paths, 
             static_pts_paths=None, R_path=None, b_path=None, zadd=None, **kwargs):
    '''
    moving_zarr_path - path to the moving_zarr to be warped
    fixed_zarr_path - path to the fixed_zarr (target)
    warped_zarr_path - path to the sink warped moving_zarr
    moving_pts_paths - list, list of numpy paths to the moving points 
    fixed_pts_paths - list, list of numpy paths to the fixed points 

    Optional:
    static_pts_paths - list, list of numpy paths to points that should remain static (should be in moving frame)
    R_path - path to the R matrix (affine transform), takes moving frame to fixed frame. If None, don't do affine transform
    b_path - path to the b matrix (translation transform)
    zadd - int, if specified, then we will add this to the warped zarr shape. Else it will be inferred from b

    **kwargs - keyword arguments for TPS_zarr_warp function 
    '''

    # affine transform parameters
    if R_path is not None:
        R = np.load(R_path); b = np.load(b_path)
        Rinv,binv = get_inverse_rigid_transform(R,b)
        if zadd is None: 
            zadd = np.maximum(b[2],0)
    else:
        Rinv = None; binv = None
    if zadd is None:
        zadd = 0
    

    # Size of the new image 
    if type(fixed_zarr_path) == str:
        z = zarr.open(fixed_zarr_path,mode='r')
        fixed_size = z.shape[:2]
    else:
        fixed_size = fixed_zarr_path[:2] 
    w = zarr.open(moving_zarr_path,mode='r')
    warped_zarr_size = (*fixed_size,int(zadd+w.shape[2])) 
    print(warped_zarr_size)



    # Construct the fixed and moving points 
    if moving_pts_paths is not None:
        fixed_pts = np.zeros((0,3))
        moving_pts = np.zeros((0,3))
        for idx in range(len(fixed_pts_paths)):
            fixed_pts_temp = np.load(fixed_pts_paths[idx])
            moving_pts_temp = np.load(moving_pts_paths[idx])
            fixed_pts = np.concatenate((fixed_pts,fixed_pts_temp),axis=0)
            moving_pts = np.concatenate((moving_pts,moving_pts_temp),axis=0)
    else:
        moving_pts = None; fixed_pts = None

    # static pts
    if static_pts_paths is not None:
        anchor_moving = np.zeros((0,3))
        for static_pts_path in static_pts_paths:
            static = np.load(static_pts_path)
            anchor_moving = np.concatenate((anchor_moving,static),axis=0)
        if Rinv is not None:
            anchor_fixed = rigid_transform(R,b,anchor_moving)
        else:
            anchor_fixed = anchor_moving.copy()

        moving_pts_total = np.concatenate((moving_pts,anchor_moving),axis=0)
        fixed_pts_total =np.concatenate((fixed_pts,anchor_fixed),axis=0)
    else:
        moving_pts_total = moving_pts; fixed_pts_total = fixed_pts 

    TPS_zarr_warp(moving_zarr_path, warped_zarr_path,
                      moving_pts_total, fixed_pts_total, warped_zarr_size=warped_zarr_size,
                      Rinv=Rinv, binv=binv, **kwargs)

## Use this to actually do grid transform of points
def grid_transform_pts(grid_path, pts_path, warped_zarr_path, 
                        inverse_transform=False,save_path=None, save_json=False):
    '''
    Transform points based on an existing grid.
    
    if inverse_transform is True,  warp points from moving to fixed  
    warped_zarr_path - path to a zarr for which we derive the shape of the image (frame of grid)
                        if tuple, then use this directly as a shape
    '''
    grid = np.load(grid_path) 
    if type(pts_path) == str:
        pts = np.load(pts_path)
    else:
        pts = pts_path # assume it's a numpy array 

    if type(warped_zarr_path) == str:
        z = zarr.open(warped_zarr_path,mode='r')
        size_ = z.shape
    else:
        size_ = warped_zarr_path 

    # This will tell us what coordinates each point on the grid is at
    linspaces = [np.linspace(0,size_[i],grid.shape[i+3]) for i in range(3)]
    if not inverse_transform:
        x_interpolator = rgi(tuple(linspaces),grid[0,0,0])
        y_interpolator = rgi(tuple(linspaces),grid[1,0,0])
        z_interpolator = rgi(tuple(linspaces),grid[2,0,0])

        coords = np.vstack((x_interpolator(pts),y_interpolator(pts),z_interpolator(pts))).T
    else:
        # Create interpolator for reverse grid 
        xs = np.expand_dims(grid[0,0,0].ravel(),1) # in the moving frame 
        ys = np.expand_dims(grid[1,0,0].ravel(),1)
        zs = np.expand_dims(grid[2,0,0].ravel(),1)
        moving_coords = np.concatenate((xs,ys,zs),axis=1)

        # The "values" at each point (moving_coords)
        X,Y,Z = np.meshgrid(linspaces[0],linspaces[1],linspaces[2],indexing='ij') 
        xs_ = X.ravel(); ys_ = Y.ravel(); zs_ = Z.ravel()

        x_lni = lni(moving_coords, xs_)
        y_lni = lni(moving_coords, ys_)
        z_lni = lni(moving_coords, zs_)
        coords = np.vstack((x_lni(pts),y_lni(pts),z_lni(pts))).T

    if save_json:
        numpy_to_json(coords, save_path[:-4]+'.json')
    if save_path is not None:
        np.save(save_path, coords)

    return coords 


############### Grid warp tools

# def resample_grid(grid_path, grid_spacing, resample_factor=(1,1,1), b_path=None,
#                     xrange=None,yrange=None,zrange=None,diffs=None,save_grid_path=None):
#     '''
#     Resample a grid for any resampling factor and FOV 

#     xrange,yrange,zrange - lists, if not None, this is the FOV from the original image we care about.
#     b_path - path to the translation vector, which will tell us if we have to add slices 
#     dxdydz - [dx,dy,dz], if not None, the initial crop coordinates of the processed original image
#     returns: resampled grid, new grid spacing 
#     '''
#     if b_path is not None:
#         b = np.load(b_path)
#         zadd = int(b[2]) 
#     else:
#         zadd = 0 

#     grid = np.load(grid_path)
#     ranges = [xrange,yrange,zrange]
#     dx,dy,dz = diffs 
#     xr = [xrange[i] - dx for i in range(2)]
#     yr = [yrange[i] - dy for i in range(2)]
#     zr = [zrange[i] - dz for i in range(2)]; zr[1] += zadd 
#     grid_spacing_new = tuple([int(np.round(grid_spacing[i]/resample_factor[i])) for i in range(3)]) # upsampled grid spacing 

#     # Need to get the relevant parts of the grid 
#     # First downsample the ranges and divide by grid spacing to get which element of grid we should focus on
#     xr = [xr[i]*resample_factor[0]/grid_spacing[0] for i in range(2)]
#     yr = [yr[i]*resample_factor[1]/grid_spacing[1] for i in range(2)]
#     zr = [zr[i]*resample_factor[2]/grid_spacing[2] for i in range(2)]

#     # Begin constructing the grid in the downsampled reference frame
#     xs = np.arange(xr[0],int(xr[1]),1)
#     ys = np.arange(yr[0],int(yr[1]),1)
#     zs = np.arange(zr[0],int(zr[1]),1)

#     xs_ = xs - int(xr[0])
#     ys_ = ys - int(yr[0])
#     zs_ = zs - int(zr[0])

#     # Get coordinates 
#     X,Y,Z = np.meshgrid(xs_,ys_,zs_)
#     xyz = np.vstack([X.ravel(), Y.ravel(),Z.ravel()]).T

#     grid_subvolume = grid[:,:,:,int(xr[0]):int(xr[1]+1),int(yr[0]):int(yr[1]+1),int(zr[0]):int(zr[1]+1)]
#     gridx = grid_subvolume[0,0,0]
#     gridy = grid_subvolume[1,0,0]
#     gridz = grid_subvolume[2,0,0]


#     newx = trilinear_interp(xyz,gridx).reshape(X.shape)
#     newy = trilinear_interp(xyz,gridy).reshape(Y.shape)
#     newz = trilinear_interp(xyz,gridz).reshape(Z.shape)
#     grid_new_untransformed = [newx,newy,newz]

#     # transform the grid to our new coordinates. Upsample --> translate 
#     grid_new = np.array([grid_new_untransformed[i]/resample_factor[i] - (ranges[i][0]-diffs[i]) for i in range(3)])
#     grid_new = np.expand_dims(np.expand_dims(grid_new,axis=1),axis=1)

#     if save_grid_path is not None:
#         np.save(save_grid_path, grid_new)

#     return grid_new, grid_spacing_new 

def crop_grid(grid_path, original_shape, new_grid_spacing, range_mode='moving',
                xrange=None,yrange=None,zrange=None,save_grid_path=None):
    '''
    Crop a grid

    range_mode - if 'moving', we crop xrange,yrange,zrange in moving frame. if 'fixed', we crop in fixed frame (grid itself)
    '''
    grid = np.load(grid_path)

    linspaces = [np.linspace(0,original_shape[i],grid.shape[i+3]) for i in range(3)]
    grid_spacing = [linspaces[i][1]-linspaces[i][0] for i in range(3)]

    if range_mode=='moving':
        ranges = [xrange,yrange,zrange]
        # compute the range in fixed frame
        xargs = np.argwhere((grid[0]>=xrange[0]-grid_spacing[0]) * (grid[0]<xrange[1]+grid_spacing[0]) *\
                        (grid[1]>=yrange[0]-grid_spacing[1]) * (grid[1]<yrange[1]+grid_spacing[1]) *\
                        (grid[2]>=zrange[0]-grid_spacing[2]) * (grid[2]<zrange[1]+grid_spacing[2]))
        args = [[xargs[:,2].min(), xargs[:,2].max()],
               [xargs[:,3].min(), xargs[:,3].max()],
                [xargs[:,4].min(), xargs[:,4].max()]]
        # print(args)

        # print(linspaces[0][args[0][0]], linspaces[0][args[0][1]])
        # print(linspaces[1][args[1][0]], linspaces[1][args[1][1]])
        # print(linspaces[2][args[2][0]], linspaces[2][args[2][1]])

        xr = [int(linspaces[0][args[0][0]]), int(linspaces[0][args[0][1]])]
        yr = [int(linspaces[1][args[1][0]]), int(linspaces[1][args[1][1]])]
        zr = [int(linspaces[2][args[2][0]]), int(linspaces[2][args[2][1]])]
        shapes = (xr,yr,zr)

        new_grid = grid[:,:,:,args[0][0]:args[0][1]+1,args[1][0]:args[1][1]+1,args[2][0]:args[2][1]+1]
        new_shape = (xr[1]-xr[0],yr[1]-yr[0],zr[1]-zr[0])
        print("New fixed image shape:",new_shape)
        print("New fixed range: x",xr[0],xr[1])
        print("New fixed range: y",yr[0],yr[1])
        print("New fixed range: z",zr[0],zr[1])



        # translate the new_grid values
        new_grid[0] -= xrange[0]; new_grid[1] -= yrange[0]; new_grid[2] -= zrange[0]

        if save_grid_path is not None:
            np.save(save_grid_path, new_grid)
        return new_grid, new_shape
    else:
        pass

    # new_linspaces = [np.linspace(0,new_shape[i],args[i][1]-args[i][0]+1) for i in range(3)]
    # initialize grid interpolator 
    # x_interpolator = rgi(tuple(new_linspaces),grid[0,0,0,args[0][0]-1:args[0][1]+1])
    # y_interpolator = rgi(tuple(new_linspaces),grid[1,0,0])
    # z_interpolator = rgi(tuple(new_linspaces),grid[2,0,0])

    

def resample_grid(grid_path, original_shape, new_grid_spacing, #resample_factor=(1,1,1),
                     xrange=None,yrange=None,zrange=None,diffs=None,save_grid_path=None):

    ## in case we need to add z_add 
    # if b_path is not None:
    #     zadd = int(np.load(b_path)[2])
    # else:
    #     zadd = 0

    grid = np.load(grid_path)

    ranges = [xrange,yrange,zrange]
    dx,dy,dz = diffs
    xr = [xrange[i] - dx for i in range(2)]
    yr = [yrange[i] - dy for i in range(2)]
    zr = [zrange[i] - dz for i in range(2)]; #zr[1] += zadd
    shapes = (xr,yr,zr)

    linspaces = [np.linspace(0,original_shape[i],grid.shape[i+3]) for i in range(3)]
    grid_spacing = [linspaces[i][1]-linspaces[i][0] for i in range(3)]
    # new_linspaces = [linspaces[i]/resample_factor[i] for i in range(3)]
    new_linspaces = [np.linspace(0,shapes[i][1],len(linspaces[i])) for i in range(3)]

    # Compute the actual "resample factor"
    resample_factor = [(linspaces[i][1]-linspaces[i][0])/(new_linspaces[i][1]-new_linspaces[i][0]) for i in range(3)]

    # initialize grid interpolator 
    x_interpolator = rgi(tuple(new_linspaces),grid[0,0,0]/resample_factor[0])
    y_interpolator = rgi(tuple(new_linspaces),grid[1,0,0]/resample_factor[1])
    z_interpolator = rgi(tuple(new_linspaces),grid[2,0,0]/resample_factor[2])
    

    # new grid
    nb_pts = (int((xr[1]-xr[0])/new_grid_spacing[0]),int((yr[1]-yr[0])/new_grid_spacing[1]),int((zr[1]-zr[0])/new_grid_spacing[2]))
    xs = np.linspace(xr[0],xr[1],nb_pts[0])
    ys = np.linspace(yr[0],yr[1],nb_pts[1])
    zs = np.linspace(zr[0],zr[1],nb_pts[2])
    X,Y,Z = np.meshgrid(xs,ys,zs, indexing='ij')


    coords = np.vstack((X.ravel(),Y.ravel(),Z.ravel())).T
    x_new = np.expand_dims((x_interpolator(coords) - xr[0]).reshape(X.shape),0)
    y_new = np.expand_dims((y_interpolator(coords) - yr[0]).reshape(X.shape),0)
    z_new = np.expand_dims((z_interpolator(coords) - zr[0]).reshape(X.shape),0)
    grid_new = np.expand_dims(np.expand_dims(np.concatenate((x_new,y_new,z_new),axis=0),1),1)

    if save_grid_path is not None:
        np.save(save_grid_path, grid_new)

    return grid_new 


def fit_rbf(affine_pts, moving_pts, smooth=0, mode='thin_plate'):
    rbf_z = Rbf(affine_pts[:, 0], affine_pts[:, 1], affine_pts[:, 2], moving_pts[:, 0], smooth=smooth, function=mode)
    rbf_y = Rbf(affine_pts[:, 0], affine_pts[:, 1], affine_pts[:, 2], moving_pts[:, 1], smooth=smooth, function=mode)
    rbf_x = Rbf(affine_pts[:, 0], affine_pts[:, 1], affine_pts[:, 2], moving_pts[:, 2], smooth=smooth, function=mode)
    return rbf_z, rbf_y, rbf_x


def rbf_transform(pts, rbf_z, rbf_y, rbf_x):
    zi = rbf_z(pts[:, 0], pts[:, 1], pts[:, 2])
    yi = rbf_y(pts[:, 0], pts[:, 1], pts[:, 2])
    xi = rbf_x(pts[:, 0], pts[:, 1], pts[:, 2])
    return np.column_stack([zi, yi, xi])


def nonrigid_transform(pts, affine_transform, rbf_z, rbf_y, rbf_x):
    affine_pts = affine_transform(pts)
    return rbf_transform(affine_pts, rbf_z, rbf_y, rbf_x)

def match_distance(pts1, pts2):
    """Calculate the distance between matches points"""
    return np.linalg.norm(pts1-pts2, axis=-1)