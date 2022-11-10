import numpy as np
from sklearn.neighbors import KDTree
from itertools import combinations
from scipy.optimize import linear_sum_assignment 
from scipy.ndimage import affine_transform 
from sklearn import linear_model, metrics 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from .rigid import rigid_transform_3D


# Module for performing feature matching


# Better way to do RANSAC
def use_ransac(moving_coords,fixed_coords,error_threshold,min_samples):
    ransac = linear_model.RANSACRegressor(residual_threshold=error_threshold,
                                              min_samples = min_samples, #max_trials=max_trials,
                                              random_state = 42)
    if min_samples is None:
        min_samples = moving_coords.shape[1]+1
        # print(min_samples)
    if len(moving_coords) >= min_samples: 
        ransac.fit(moving_coords,fixed_coords)

        # Get the residuals 
        y_pred = ransac.predict(moving_coords)
        #print("RANSAC mean squared error: %.2f" %metrics.mean_absolute_error(y_pred, fixed_coords))

        inlier_mask = ransac.inlier_mask_ 
        moving_coords_ransac = moving_coords[inlier_mask]
        fixed_coords_ransac = fixed_coords[inlier_mask] 
    else:
        #print(min_samples)
        moving_coords_ransac = np.zeros((0,3))
        fixed_coords_ransac = np.zeros((0,3))
    return moving_coords_ransac,fixed_coords_ransac


def apply_ransac_v2(moving_pts_paths, fixed_pts_paths, moving_save_path=None, fixed_save_path=None, points_idxs_to_evaluate=None,
                    error_threshold=30, min_samples=1, radius=500, voxel_size=(1,1,1)):
    '''
    points_idxs_to_evaluate - list of points idxs to actually apply ransac to. If None, use all points
    '''

    moving_coords = np.zeros((0,3),dtype='int')
    fixed_coords = np.zeros((0,3),dtype='int')

    if points_idxs_to_evaluate is not None:
        point_idxs_other = set(tuple(np.arange(len(moving_pts_paths)))).symmetric_difference(
                                set(tuple(points_idxs_to_evaluate))
        )
        moving_coords_other = np.zeros((0,3),dtype='int')
        fixed_coords_other = np.zeros((0,3),dtype='int')
        for idx in points_idxs_to_evaluate:
            moving_pt_path = moving_pts_paths[idx]
            if type(moving_pt_path)==str:
                moving_pt_temp = np.load(moving_pt_path)
            else:
                moving_pt_temp = moving_pt_path 
            moving_coords = np.concatenate((moving_coords,moving_pt_temp),axis=0)
            fixed_pt_path = fixed_pts_paths[idx]
            if type(fixed_pt_path)==str:
                fixed_pt_temp = np.load(fixed_pt_path)
            else:
                fixed_pt_temp = fixed_pt_path 
            fixed_coords = np.concatenate((fixed_coords,fixed_pt_temp),axis=0)
        for idx in point_idxs_other:
            moving_pt_path = moving_pts_paths[idx]
            if type(moving_pt_path)==str:
                moving_pt_temp = np.load(moving_pt_path)
            else:
                moving_pt_temp = moving_pt_path 
            moving_coords_other = np.concatenate((moving_coords_other,moving_pt_temp),axis=0)
            fixed_pt_path = fixed_pts_paths[idx]
            if type(fixed_pt_path)==str:
                fixed_pt_temp = np.load(fixed_pt_path)
            else:
                fixed_pt_temp = fixed_pt_path 
            fixed_coords_other = np.concatenate((fixed_coords_other,fixed_pt_temp),axis=0)
        moving_coords_total = np.concatenate((moving_coords,moving_coords_other),axis=0)
        fixed_coords_total = np.concatenate((fixed_coords,fixed_coords_other),axis=0)
        
        moving_coords_ransac=[];fixed_coords_ransac=[]
        for ix,moving_coord in tqdm(enumerate(moving_coords)):
            fixed_coord = fixed_coords[ix]
            moving_dists = np.sqrt(np.square((moving_coords_total[:,0]-moving_coord[0])*voxel_size[0])+\
                                np.square((moving_coords_total[:,1]-moving_coord[1])*voxel_size[1]))
            moving_coords_local = moving_coords_total[moving_dists<=radius]
            fixed_coords_local = fixed_coords_total[moving_dists<=radius]
            moving_coords_inlier,_ = use_ransac(moving_coords_local,fixed_coords_local,error_threshold,min_samples)

            if moving_coord in moving_coords_inlier:
                moving_coords_ransac.append(moving_coord)
                fixed_coords_ransac.append(fixed_coord)
                
        moving_coords_ransac = np.array(moving_coords_ransac); 
        moving_coords_ransac = np.concatenate((moving_coords_ransac,moving_coords_other),axis=0)
        fixed_coords_ransac = np.array(fixed_coords_ransac)
        fixed_coords_ransac = np.concatenate((fixed_coords_ransac,fixed_coords_other),axis=0)
    else:
        for moving_pt_path in moving_pts_paths:
            if type(moving_pt_path)==str:
                moving_pt_temp = np.load(moving_pt_path)
            else:
                moving_pt_temp = moving_pt_path 
            moving_coords = np.concatenate((moving_coords,moving_pt_temp),axis=0)
        for fixed_pt_path in fixed_pts_paths:
            if type(fixed_pt_path)==str:
                fixed_pt_temp = np.load(fixed_pt_path)
            else:
                fixed_pt_temp = fixed_pt_path 
            fixed_coords = np.concatenate((fixed_coords,fixed_pt_temp),axis=0)

        moving_coords_ransac=[];fixed_coords_ransac=[]
        for ix,moving_coord in tqdm(enumerate(moving_coords)):
            fixed_coord = fixed_coords[ix]
            moving_dists = np.sqrt(np.square((moving_coords[:,0]-moving_coord[0])*voxel_size[0])+\
                                np.square((moving_coords[:,1]-moving_coord[1])*voxel_size[1]))
            moving_coords_local = moving_coords[moving_dists<=radius]
            fixed_coords_local = fixed_coords[moving_dists<=radius]
            moving_coords_inlier,_ = use_ransac(moving_coords_local,fixed_coords_local,error_threshold,min_samples)
            if moving_coord in moving_coords_inlier:
                moving_coords_ransac.append(moving_coord)
                fixed_coords_ransac.append(fixed_coord)
                                              
        moving_coords_ransac = np.array(moving_coords_ransac);
        fixed_coords_ransac = np.array(fixed_coords_ransac)
    if moving_save_path is not None:
        np.save(moving_save_path,moving_coords_ransac)
    if fixed_save_path is not None:
        np.save(fixed_save_path,fixed_coords_ransac)

    return moving_coords_ransac, fixed_coords_ransac 


def compute_spatial_descriptor_moving(coords, moving_anchors, fixed_anchors, num_nn, use3d=True, voxel_size=(1,1,1),
                                      R=None, b=None,search_radius=500,transform_type='translation',min_anchor_nns=10):
    '''
    use3d - if True, cmpute spatial feature descriptor in 3D. Else, 2D. (note: does not apply to anchor search_radius)
    search_radius: int, lateral 2D search radius about each coord to search for anchors to compute transformation
    transform_type: 'translation' or 'rigid', what kind of transform to calculate and apply to transform
    '''
    
    if R is not None:
        if type(R) == str:
            R = np.load(R)
        if type(b) == str:
            b = np.load(b)
    coords = np.transpose(np.matmul(R,np.transpose(coords)) + np.expand_dims(b,1))
    moving_anchors = np.transpose(np.matmul(R,np.transpose(moving_anchors)) + np.expand_dims(b,1))
    coords_ = coords.copy().astype('float')
    coords_[:,0] *= voxel_size[0]; coords_[:,1] *= voxel_size[1]; coords_[:,2]*=voxel_size[2]
    moving_anchors_ = moving_anchors.copy().astype('float')
    moving_anchors_[:,0] *= voxel_size[0]; moving_anchors_[:,1] *= voxel_size[1]; moving_anchors_[:,2]*= voxel_size[2]
    
    # Anchor nearest neighbros 
    kdt = KDTree(moving_anchors_[:,:2],leaf_size=30)
    dists,inds = kdt.query(coords_[:,:2],k=100)
    
    
    if not use3d:
        n = 2 
    else:
        n = 3
    descriptor = np.zeros((coords.shape[0],num_nn*n),dtype='float')
    new_coords = np.zeros(coords.shape)
    
    
    for idx,_ in tqdm(enumerate(coords)):
        inds_local = inds[idx][dists[idx]<=search_radius]
        if len(inds_local) < min_anchor_nns:
            inds_local = inds[idx][:min_anchor_nns]
        moving_anchors_local = moving_anchors[inds_local]
        fixed_anchors_local = fixed_anchors[inds_local]

        if transform_type=='rigid':
            Rn,bn = rigid_transform_3D(np.transpose(moving_anchors_local[:,:2]), 
                                                 np.transpose(fixed_anchors_local[:,:2]))
            R_local = np.eye(3); R_local[:2,:2] = Rn
            b_local = np.zeros((3,1)); b_local[:2] = bn
        else:
            R_local = np.eye(3)
            b_local = np.array([[np.mean(fixed_anchors_local[:,0]) - np.mean(moving_anchors_local[:,0])],
                                [np.mean(fixed_anchors_local[:,1]) - np.mean(moving_anchors_local[:,1])],
                                0])


        coords_transformed = np.transpose(np.matmul(R_local,np.transpose(coords)) + b_local)
        coords_transformed_ = coords_transformed.copy().astype('float')
        coords_transformed_[:,0]*=voxel_size[0]; coords_transformed_[:,1]*=voxel_size[1]; coords_transformed_[:,1]*=voxel_size[1]
        
        
        kdt_local = KDTree(coords_transformed_[:,:n], leaf_size=30) 
        _, inds_local = kdt_local.query(coords_transformed_[idx:idx+1,:n], k=num_nn+1)
        inds_local = inds_local.ravel()
        for j in range(1,num_nn+1):
            descriptor[idx,(j-1)*n:n*j] = coords_transformed_[inds_local[j],:n]-coords_transformed_[idx,:n]
        new_coords[idx] = coords_transformed[idx]
    
    return new_coords, descriptor


def compute_spatial_descriptor(coords, num_nn, use3d=True, voxel_size=(1,1,1)):
    '''
    Computes feature descriptor containing the relative 2D coordinates of the nearest neighbors to a given point.
        
    Inputs:
    coords - n*3 array containing all of the coordinates of the endpoints 
    num_nn - the number of spatial nearest neighbors to include in the descriptor. 
    
    Outputs:
    descriptor - n*(2*num_nn) array containing the descriptor for each point 
    '''
    
    coords_ = coords.copy().astype('float')
    coords_[:,0] *= voxel_size[0]; coords_[:,1] *= voxel_size[1]; coords_[:,2]*=voxel_size[2]
    
    if not use3d:
        n = 2 
    else:
        n = 3
    kdt = KDTree(coords_[:,:n], leaf_size=30) 
    _, inds = kdt.query(coords_[:,:n], k=num_nn+1)
    descriptor = np.zeros((coords.shape[0],num_nn*n),dtype='float')
    for i in range(coords.shape[0]):
        for j in range(1,num_nn+1):
            descriptor[i,(j-1)*n:n*j] = coords_[inds[i,j],:n]-coords_[i,:n]

    return descriptor 

    
def compute_feature_matches(moving_coords, moving_descriptor, fixed_coords, fixed_descriptor, 
                            use3d=True, search_radius=50, ratio_thresh=0.85, num_nn_used=3, return_nn=False,
                            voxel_size=(1,1,1)):
    '''
    Computes the good matches between a set of coordinates based on the spatial descriptor. 
    
    Inputs:
    moving_coords - n*3 array of all of the moving coordinates 
    moving_descriptor - n*(num_nn*3) array of the spatial descriptor for all moving coordinates 
    fixed_coords - m*3 array of all fixed coordinates 
    fixed_descriptor - m*(num_nn*3) array of the spatial descriptor for all fixed coordinates 
    
    use3d - bool, if True, then calculate feautre descriptor based on 3D. Otherwise, use 2D 
    search_radius - int, pixel radius in which we search for possible matches for a given coordinate 
    ratio_thresh - float, the maximum acceptable distance ratio between the best match and 2nd-best match 
    num_nn_used - int, number of coordinates to use in computing feature distance. Must be less than num_nn 
    return_nn - bool, if True, returns the indices of the nearest neighbors that are matched (default: False)
    
    Outputs:
    moving_coords_matched - array of matched moving coordinates (3D)
    moving_descriptor_matched - array of matched moving feature 
    fixed_coords_matched 
    fixed_descriptor_matched 
    
    ''' 
    # global n 
    n = 3
    if use3d:
        dim = 3 
    else:
        dim = 2


    # For filtering out points that are too far away 
    moving_coords_ = moving_coords.copy().astype('float')
    moving_coords_[:,0] *= voxel_size[0]; moving_coords_[:,1] *= voxel_size[1]; moving_coords_[:,2] *= voxel_size[2]
    fixed_coords_ = fixed_coords.copy().astype('float')
    fixed_coords_[:,0] *= voxel_size[0]; fixed_coords_[:,1] *= voxel_size[1]; fixed_coords_[:,2] *= voxel_size[2]
    kdt = KDTree(moving_coords_[:,:dim])
    dist, inds = kdt.query(fixed_coords_[:,:dim], k=30) 
    
    num_only_one = 0 # number of points that only had one possibility in other image 
    num_both_nn = 0 
    
    # also record the indices 
    moving_inds_matched = []; fixed_inds_matched = [] 
    moving_coords_matched=[]; moving_descriptor_matched=[]; fixed_coords_matched=[]; fixed_descriptor_matched=[]
    moving_nns_matched=np.zeros((0,3),dtype='int'); fixed_nns_matched=np.zeros((0,3),dtype='int')

    # Iterate through all of the fixed coordinates 
    for i in tqdm(range(fixed_coords.shape[0])):
        candidates = inds[i][dist[i,:]<=search_radius]
        ffeat = fixed_descriptor[i]
        
        if len(candidates)>0:
            cand_feats = moving_descriptor[candidates]
            
            # Minimize the distance between features 
            Di = []
            for mfeat in cand_feats:
                if num_nn_used > len(ffeat)/n:
                    raise RuntimeError('Num_nn_used (found %d) should be less than number of nearest neighbors \
                                        in feature descriptor (%d)'%(num_nn_used, len(ffeat)/n))
                else:
                    best_dist = _compute_distance_between_features(mfeat, ffeat, num_nn_used)
                    Di.append(best_dist) 
                    
                
            inds_i = np.argsort(Di) # sorted indices 
            if len(Di) > 1:
                thresh = Di[inds_i[0]]/Di[inds_i[1]]
                num_fwd_only = 0
            elif len(Di) == 1: 
                thresh = -1
                num_fwd_only = 1 # keeping track of whether there is only one on the fwd pass 
            
            # If we found a good fwd match, then we perform the backward pass 
            if thresh <= ratio_thresh:
                moving_coords_cand = moving_coords[candidates,:dim][inds_i[0]]
                moving_descriptor_cand = cand_feats[inds_i[0]]
                
                # Get the physical nearest neighbors 
                kdt2 = KDTree(fixed_coords_[:,:dim]) 
                moving_coords_cand = moving_coords_cand.reshape(1,-1)
                moving_coords_cand_ = moving_coords_cand.copy().astype('float')
                moving_coords_cand_[:,0]*=voxel_size[0]; moving_coords_cand_[:,1]*=voxel_size[1]; 
                if use3d:
                    moving_coords_cand_[:,2]*=voxel_size[2]
                dist2, inds2 = kdt2.query(moving_coords_cand_, k=30)
                candidates_r = inds2[dist2 <= search_radius]
                cand_feats_r = fixed_descriptor[candidates_r]
                
                Dr = []
                for ffeatr in cand_feats_r:
                    best_dist = _compute_distance_between_features(moving_descriptor_cand, ffeatr, num_nn_used)
                    Dr.append(best_dist)
                
                inds_r = np.argsort(Dr)
                if len(Dr) > 1:
                    thresh_r = Dr[inds_r[0]]/Dr[inds_r[1]]
                    
                    more_than_one = 1*(1-num_fwd_only)
                elif len(Dr) == 1: 
                    thresh_r = -1
                    
                    # Just for bookkeeping / troubleshooting 
                    more_than_one = 0
                    num_only_one +=  num_fwd_only 
                    
                if thresh_r <= ratio_thresh and \
                   np.prod(fixed_coords[candidates_r][inds_r[0]] == fixed_coords[i]) > 0:
                    num_both_nn += more_than_one 
                    moving_coords_matched.append(moving_coords[candidates][inds_i[0]])
                    moving_descriptor_matched.append(moving_descriptor_cand)
                    fixed_coords_matched.append(fixed_coords[i])
                    fixed_descriptor_matched.append(fixed_descriptor[i]) 
                    
                    if return_nn:
                        # Use the saved descriptors to find out the best nn's 
                        _, fixed_nns, move_nns = _compute_distance_between_features(moving_descriptor_cand,
                                                                                    fixed_descriptor[i],
                                                                                    num_nn_used, return_nn) 
                        fixed_nns += fixed_coords[i] 
                        move_nns += moving_coords[candidates][inds_i[0]]
                        fixed_nns_matched = np.vstack((fixed_nns_matched,fixed_nns))
                        moving_nns_matched = np.vstack((moving_nns_matched,move_nns))
                        
    moving_coords_matched = np.asarray(moving_coords_matched)
    moving_descriptor_matched = np.asarray(moving_descriptor_matched)
    fixed_coords_matched = np.asarray(fixed_coords_matched)
    fixed_descriptor_matched = np.asarray(fixed_descriptor_matched) 
    
    ## The below is only to see how many nearest neighbors were forced or actually prominent
    # print('Both:',num_both_nn)
    # print('One:',num_only_one)
    return (moving_coords_matched, moving_descriptor_matched,
            fixed_coords_matched, fixed_descriptor_matched,
            moving_nns_matched, fixed_nns_matched)
    
    
def _compute_distance_between_features(mfeat, ffeat, num_nn_used, return_nn=False):
    '''
    Computes the minimal distance between two feature vectors
    
    Inputs:
    mfeat, ffeat - arrays containing the descriptors
    num_nn_used - int, the number of nearest neighbors to use 
    return_nn - bool, when True, also return the nearest neighbors of best matches 
    
    Outputs:
    best_dist - float, minimal distance between vectors 
    fixed_nns - coords of the fixed nearest neighbors 
    moving_nns - coords of the moving nearest neighbors 
    '''
    ## TODO: make this so we also have the option to return the nearest neihbors
    
    #global n # (x,y) or (x,y,z) coordinates used to compute distances 
    n = 3 # global variables are slow 
    num_nn = len(ffeat) // n
    
    # Compute the distance between every point 
    # fixed are the rows, and moving are the columns 
    cost_matrix = np.zeros((num_nn,num_nn))
    for i in range(num_nn):
        for j in range(num_nn):
            # We square each term so we can compute the distance between features correctly 
            cost_matrix[i,j] = np.linalg.norm(ffeat[i*n:n*i+2] - mfeat[j*n:n*j+2])**2
    
    # Use Hungarian algorithm to compute the best case
    min_dist = np.inf
    for combo in combinations(range(num_nn), num_nn_used):
        row_ind, col_ind = linear_sum_assignment(cost_matrix[combo,:])
        # dist = np.linalg.norm(cost_matrix[row_ind,col_ind])
        dist = np.sqrt(np.sum(cost_matrix[combo,:][row_ind,col_ind]))
        if dist < min_dist:
            min_dist = dist 
            if return_nn: 
                # we also want to keep track of which combo gives the best distance 
                fixed_nns = np.asarray([ffeat[combo[v]*n:combo[v]*n+n] for v in range(num_nn_used)]) 
                moving_nns = np.asarray([mfeat[col_ind[v]*n:col_ind[v]*n+n] for v in range(num_nn_used)])
    if return_nn:
        return min_dist, fixed_nns, moving_nns
    else:
        return min_dist 
    
    
    
def apply_ransac(moving_coords, fixed_coords, moving_descriptor=None, fixed_descriptor=None,
                 error_threshold=50, use_3d=True, use_local_ransac=False, **kwargs):
    '''
    Uses RANSAC to evaluate which correspondences are good and which are bad.
    
    Inputs:
    moving_coords, fixed_coords - arrays of coordinates 
    moving_descriptor, fixed_descriptor - arrays of the descriptors 
    
    error_threshold - float, RANSAC error threshold for affine transform. The larger the more lenient
    use_3d - bool, if True, we use RANSAC on the 3D coordinates. if False, use 2D 
    use_local_ransac - bool, if True, we use RANSAC locally on tiles. 
    
    kwargs: 
    min_samples - int, minimum number of samples used to generate model in RANSAC 
    max_trials - int, maximum number of RANSAC trials 
    (only used when use_local_ransac is True) 
    num_x_tiles, num_y_tiles - int, number of tiles in x and y direction to perform ransac
    overlap - int, number of pixels used for overlap between tiles when looking
    size_image - (int, int) 2D size of the image 
    min_matches - int, the minimum number of good correspondences in each tile to not be skipped 
    verbose - bool, if True, will print statements 
    
    Outputs:
    moving_coords_ransac, fixed_coords_ransac - coordinates of matched points 
    '''
    
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose = True # default is True 
    if 'min_samples' in kwargs: 
        min_samples = kwargs['min_samples']
    else:
        min_samples = 12
    if 'apply_affine_transform' in kwargs:
        apply_affine_transform = kwargs['apply_affine_transform']
        if apply_affine_transform is False:
            affine_matrix = None
            affine_intercept = None 
    else:
        apply_affine_transform = False 
        affine_matrix = None
        affine_intercept = None 

    if not use_local_ransac:
        ## Global ransac 
        ransac = linear_model.RANSACRegressor(residual_threshold=error_threshold,
                                              min_samples = min_samples, #max_trials=max_trials,
                                              random_state = 42)
        if not use_3d:
            ransac.fit(moving_coords[:,:2], fixed_coords[:,:2])
        else:
            ransac.fit(moving_coords, fixed_coords)

        # Get the residuals 
        y_pred = ransac.predict(moving_coords)
        if verbose:
            print("RANSAC mean squared error: %.2f" %metrics.mean_absolute_error(y_pred, fixed_coords))

        ## This part is for applying a global affine transform before the nonrigid deformation
        if apply_affine_transform:
            affine_matrix = ransac.estimator_.coef_
            affine_intercept = ransac.estimator_.intercept_        
            #print(affine_matrix, affine_intercept)


        inlier_mask = ransac.inlier_mask_ 
        moving_coords_ransac = moving_coords[inlier_mask]
        fixed_coords_ransac = fixed_coords[inlier_mask] 
        if moving_descriptor is not None:
            moving_descriptor_ransac = moving_descriptor[inlier_mask] 
            fixed_descriptor_ransac = fixed_descriptor[inlier_mask]
    else:
        ## Local ransac 
        if 'num_x_tiles' in kwargs:
            num_x_tiles = kwargs['num_x_tiles']
        else:
            num_x_tiles = 4
        if 'num_y_tiles' in kwargs:
            num_y_tiles = kwargs['num_y_tiles']
        else:
            num_y_tiles = 4
        if 'overlap' in kwargs:
            overlap = kwargs['overlap']
        else:
            overlap = 0
        if 'size_image' in kwargs:
            size_image = kwargs['size_image']
        else:
            print('Please enter an image size.')
        if 'min_matches' in kwargs:
            min_matches = kwargs['min_matches']
        else:
            min_matches = 3
        
        x_bounds = _compute_bounds(num_x_tiles, overlap, size_image[0])
        y_bounds = _compute_bounds(num_y_tiles, overlap, size_image[1])
        
        moving_coords_ransac = np.zeros((1,3))
        fixed_coords_ransac = np.zeros((1,3))
        if moving_descriptor is not None:
            moving_descriptor_ransac = np.zeros((1,moving_descriptor.shape[1]))
            fixed_descriptor_ransac = np.zeros((1,fixed_descriptor.shape[1]))
            
        for i in range(num_x_tiles):
            for j in range(num_y_tiles):
                fix_where = (fixed_coords[:,0]>=x_bounds[i,0]) * (fixed_coords[:,0]<x_bounds[i,1]) *\
                            (fixed_coords[:,1]>=y_bounds[j,0]) * (fixed_coords[:,1]<y_bounds[j,1])
                fix = fixed_coords[fix_where]
                move = moving_coords[fix_where]
                if moving_descriptor is not None:
                    move_descriptor = moving_descriptor[fix_where]
                    fix_descriptor = fixed_descriptor[fix_where] 
                if np.sum(fix_where) < min_matches:
                    if verbose:
                        print('Not enough matches in in tile (%d,%d), skipping...'%(i,j))
                else:
                    ransac = linear_model.RANSACRegressor(residual_threshold=error_threshold,
                                              #min_samples = min_samples, #max_trials=max_trials,
                                              random_state = 42)
                    if move.shape[0] > 3:
                        if not use_3d:
                            ransac.fit(move[:,:2], fix[:,:2])
                        else:
                            ransac.fit(move, fix)
                            
                        inlier_mask = ransac.inlier_mask_ 
                        
                        # Add best inliers to a list 
                        moving_coords_ransac = np.concatenate((moving_coords_ransac,move[inlier_mask]),axis=0)
                        fixed_coords_ransac = np.concatenate((fixed_coords_ransac,fix[inlier_mask]),axis=0)
                        if moving_descriptor is not None:
                            moving_descriptor_ransac = np.concatenate((moving_descriptor_ransac,move_descriptor[inlier_mask]),axis=0)
                            fixed_descriptor_ransac = np.concatenate((fixed_descriptor_ransac,fix_descriptor[inlier_mask]),axis=0)
                        if verbose:
                            print('Number of matches found:', len(move[inlier_mask]))
                            print('Finished filtering tile (%d,%d)'%(i,j))
                    else:
                        if verbose:
                            print("min_samples = %d, which is greater than the n_samples = %d"%(min_samples, move.shape[0]))
        # We have to deal with when one point has two corresponding matches in the other image for local 
        moving_coords_ransac, minds = np.unique(moving_coords_ransac[1:], axis=0, return_index=True)
        #fixed_coords_ransac, finds = np.unique(fixed_coords_ransac[1:], axis=0, return_index=True) 
        fixed_coords_ransac = fixed_coords_ransac[1:][minds]
        
        if moving_descriptor is not None:
            moving_descriptor_ransac = moving_descriptor_ransac[1:][minds]
            fixed_descriptor_ransac = fixed_descriptor_ransac[1:][minds]
            
            
    if moving_descriptor is None:
        moving_descriptor_ransac = None 
    if fixed_descriptor is None:
        fixed_descriptor_ransac = None 
        
    if apply_affine_transform is False:   
        return (moving_coords_ransac.astype('int'), fixed_coords_ransac.astype('int'), 
                moving_descriptor_ransac, fixed_descriptor_ransac) 
    else:
        return (moving_coords_ransac.astype('int'), fixed_coords_ransac.astype('int'), 
                moving_descriptor_ransac, fixed_descriptor_ransac, affine_matrix, affine_intercept)

    
    
def _compute_bounds(num_tiles, overlap, total_dim_size):
    '''
    Computes the bounds given a number of tiles and overlap and returns the start/end points.
    
    Inputs:
    num_tiles - int, number of tiles 
    overlap - int, umber of pixel overlap between tiles
    total_dim_size - the total size of the dimension 
    
    Outputs:
    bounds - (num_tiles*2) array containing the start and end points of each tile. 
    '''
    
    tile_size = int(np.ceil((total_dim_size+(num_tiles-1)*overlap)/num_tiles))
    bounds = np.ones((num_tiles,2),dtype='uint16')
    for i in range(num_tiles):
        if i != 0:
            bounds[i,0] = bounds[i-1,1] - overlap 
        if i != num_tiles-1:
            bounds[i,1] = bounds[i,0] + tile_size 
        else:
            bounds[i,1] = total_dim_size 

    return bounds 


# Function to include nearest neighbors also 

def filter_combine_nns(bcoords_nn, tcoords_nn, bcoords_matched, tcoords_matched):
    '''
    A function to combine the nearest neighbors of "good" matches with the good matches themselves 

    Inputs:
    bcoords_nn - N*3 array of bottom nearest neighbors coordaintes 
    tcoords_nn - N*3 array of top nearest neighbors coordinates 
    bcoords_matched - N*3 array of matched bottom coordinates 
    tcoords_matched - N*3 array of matched top coordinates 
    '''

    # a) Get rid of nearest neighbors that have more than one correspondence 
    unique_bcoords_nn = np.unique(bcoords_nn,axis=0)
    unique_tcoords_nn = np.unique(tcoords_nn,axis=0)

    # FIrst concatenate the two 
    combined_nns = np.hstack((bcoords_nn, tcoords_nn))
    unique_nn_matches = np.unique(combined_nns, axis=0)

    bunique_nn_matches = unique_nn_matches.copy() # keep track of bottom nns
    for bc in unique_bcoords_nn:
        bargs = np.argwhere((bunique_nn_matches[:,0]==bc[0])*(bunique_nn_matches[:,1]==bc[1])*(bunique_nn_matches[:,2]==bc[2]))
        if len(bargs)>1:
            bunique_nn_matches = np.delete(bunique_nn_matches, list(bargs.reshape(-1,)), axis=0)

    tunique_nn_matches = unique_nn_matches.copy() # keep track of top nn's 
    for tc in unique_tcoords_nn:
        targs = np.argwhere((tunique_nn_matches[:,3]==tc[0])*(tunique_nn_matches[:,4]==tc[1])*(tunique_nn_matches[:,5]==tc[2]))
        if len(targs)>1:
            tunique_nn_matches = np.delete(tunique_nn_matches, list(targs.reshape(-1,)), axis=0)

            
    print("Number of NN's (first pass):",len(bunique_nn_matches))
    print("Number of NN's (first pass):",len(tunique_nn_matches))

    # Now we need to switch and delete repeats in the other list 
    for bc in unique_bcoords_nn:
        bargs = np.argwhere((tunique_nn_matches[:,0]==bc[0])*(tunique_nn_matches[:,1]==bc[1])*(tunique_nn_matches[:,2]==bc[2]))
        if len(bargs)>1:
            tunique_nn_matches = np.delete(tunique_nn_matches, list(bargs.reshape(-1,)), axis=0)

    for tc in unique_tcoords_nn:
        targs = np.argwhere((bunique_nn_matches[:,3]==tc[0])*(bunique_nn_matches[:,4]==tc[1])*(bunique_nn_matches[:,5]==tc[2]))
        if len(targs)>1:
            bunique_nn_matches = np.delete(bunique_nn_matches, list(targs.reshape(-1,)), axis=0)

            
    print("Number of NN's (second pass):",len(bunique_nn_matches))
    print("Number of NN's (second pass):",len(tunique_nn_matches))


    # Now finally synthesize the two together to get the final list. 
    final_nn_matches = np.unique(np.vstack((bunique_nn_matches, tunique_nn_matches)),axis=0)
    print("Final number of matched NN's:",final_nn_matches.shape)

    # Confirm that we now only have unique correspondences
    if np.unique(final_nn_matches[:,:3],axis=0).shape[0] != final_nn_matches.shape[0]\
        or np.unique(final_nn_matches[:,3:],axis=0).shape[0] != final_nn_matches.shape[0]:
        print("Some double correspondences still remain, please go back and check why.")


    # Now we combine these with the matched points, but we have to filter
    # out the repeats (again)

    total_matches = np.hstack((bcoords_matched,tcoords_matched))
    total_matches_w_nns = np.vstack((total_matches, final_nn_matches))
    # total_matches_w_nns = np.unique(total_matches_w_nns, axis=0)

    for coord in total_matches:
        # find the indices where the NN's contain one set of coordinates
        # matching, but corresponds with the incorrect coords
        bargs = np.argwhere(((final_nn_matches[:,0]==coord[0])*\
                             (final_nn_matches[:,1]==coord[1])*\
                             (final_nn_matches[:,2]==coord[2]))*\
                            ((final_nn_matches[:,3]!=coord[3])+\
                             (final_nn_matches[:,4]!=coord[4])+\
                             (final_nn_matches[:,5]!=coord[5])))
        targs = np.argwhere(((final_nn_matches[:,3]==coord[3])*\
                             (final_nn_matches[:,4]==coord[4])*\
                             (final_nn_matches[:,5]==coord[5]))*\
                            ((final_nn_matches[:,0]!=coord[0])+\
                             (final_nn_matches[:,1]!=coord[1])+\
                             (final_nn_matches[:,2]!=coord[2])))
        args_to_delete = np.hstack((bargs.reshape(-1,),targs.reshape(-1,)))
        total_matches_w_nns = np.delete(total_matches_w_nns, 
                                        list(args_to_delete), axis=0)

    # Now we have the final list
    total_matches_w_nns = np.unique(total_matches_w_nns, axis=0)
    print("Total matched points (including NNs):", total_matches_w_nns.shape)
    bcoords_matched_w_nns = total_matches_w_nns[:,:3]
    tcoords_matched_w_nns = total_matches_w_nns[:,3:]
    return bcoords_nn, tcoords_nn 

###################

def find_corresponding_indices(ransaced_pts, og_pts, num_workers=1):
    '''
    Used for finding corresponding indices from RANSAC points 
    '''
    pts = og_pts.astype('int')
    indices = []
    for pt in ransaced_pts:
        ind = np.argwhere((pts[:,0]==pt[0]) * (pts[:,1]==pt[1]) * (pts[:,2]==pt[2]))[0]
        indices.append(ind)
    return indices
