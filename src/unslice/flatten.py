from . import utils
from .registration.utils import fit_surface 
from . import IO as io 
from tqdm import tqdm 
import numpy as np 
from scipy.spatial import distance, Delaunay 
import zarr 
import open3d as o3d
import subprocess
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import trimesh
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from .registration.rigid import * 
from sklearn.cluster import KMeans
from scipy.interpolate import griddata
from PIL import Image, ImageDraw


bff_global_path = '/mnt/storage/unslice/uvmap/boundary-first-flattening/build/bff-command-line' # where bff executable is located 

def flatten(eps_path, eps_uv_path, zcoord=None, plot=False, alpha=None):
    '''
    Overall function for flattening using UV map
    '''
    eps = np.load(eps_path)
    mesh_path = eps_path[:-4]+'_mesh.obj'
    uvmap_mesh_path = eps_path[:-4]+'_mesh_uv.obj'

    # 1. Create the surface mesh based on the endpoints first 
    faces = pcloud_to_triangles(eps,plot=False,alpha=alpha)
    mesh = create_surface_mesh(eps, faces, mesh_path)

    # 2. Boundary first flattening
    uv_map(mesh_path, uvmap_mesh_path)

    if zcoord is None:
        zcoord = eps[:,2].mean()

    # 3. Get UVmap from the obj
    eps_uv, eps = extract_uvmap_from_obj(uvmap_mesh_path, zcoord)

    # 4. Save the flattened coordinates 
    np.save(eps_uv_path, eps_uv)

    if plot:
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.scatter(eps_uv[:,0],eps_uv[:,1],eps_uv[:,2],antialiased=True, alpha=0.1)
        ax.scatter(eps[:,0],eps[:,1],eps[:,2],antialiased=True,alpha=0.1)
        ax.set_zlim3d(eps[:,2].min()-200,eps[:,2].max()+200)

    return eps_uv 


def align_uv_maps(top_eps, top_eps_uv, bot_eps, bot_eps_uv, thickness_filter=False, nns=10, n_std=2, plot=False):
    '''
    Rigidly align UV maps in lateral space for a given tissue, 
    and then rigidly laterally align the UV maps to the tissue. 

    Computes the thickness distribution of the tissue, and then (if not False), apply a thickness filter 
    (Get rid of points that are more than n_std Std. Deviations away from the mean thickness).


    '''

    # First apply thickness filter both ways (or just compute thickness) 
    top_eps_flat, top_eps_uv = compute_thickness(top_eps, top_eps_uv, bot_eps, thickness_filter=thickness_filter, nns=nns, n_std=n_std, plot_hist=False)
    bot_eps_flat, bot_eps_uv = compute_thickness(bot_eps, bot_eps_uv, top_eps, thickness_filter=thickness_filter, nns=nns, n_std=n_std, plot_hist=False)
    
    # Compute the new thickness and add to the flattened bottom slab endpoints 
    kdt = KDTree(bot_eps_flat[:,:2], leaf_size=30) 
    _, inds = kdt.query(top_eps_flat[:,:2], k=nns)
    thicknesses = []
    for idx in range(len(inds)):
        thickness = bot_eps_flat[inds[idx]][:,2].mean() - top_eps_flat[idx,2]
        thicknesses.append(thickness)
    thicknesses = np.array(thicknesses)
    print("New mean thickness:",thicknesses.mean())

    bot_centroids = np.mean(bot_eps_flat[inds],axis=1)
    bot_uv_centroids_rearranged = np.mean(bot_eps_uv[inds],axis=1)

    # Add thickness to bot_eps_uv
    bot_eps_uv[:,2] = thicknesses.mean()+top_eps_uv[0,2]

    if plot:
        plt.figure()
        plt.hist(thicknesses)
        plt.xlabel('Thickness (pixel)')
        plt.ylabel('Frequency')
        plt.show()

    # Now we rigidly transform (laterally) the UV maps 
    R,b = rigid_transform_3D(np.transpose(bot_uv_centroids_rearranged[:,:2]), np.transpose(top_eps_uv[:,:2]))
    # new_pts = np.transpose(np.matmul(R,np.transpose(bot_uv_centroids_rearranged[:,:2])) + b)

    # transform the actual points
    new_points = np.transpose(np.matmul(R,np.transpose(bot_eps_uv[:,:2]))+b)
    bot_eps_uv = np.concatenate((new_points,bot_eps_uv[:,2:3]),axis=1)

    if plot:
        fig = plt.figure()
        plt.title('After UV-UV rigid alignment')
        ax = fig.add_subplot(1,1,1)
        ax.scatter(top_eps_uv[:,0],top_eps_uv[:,1],antialiased=True, alpha=0.05, color='b')
        ax.scatter(bot_eps_uv[:,0],bot_eps_uv[:,1],antialiased=True,alpha=0.03,color='r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(['Top UV','Bot UV'])

    # Now we rigidly align the UV maps to the tissue 
    target_pts = np.transpose(np.concatenate((bot_eps_flat[:,:2],top_eps_flat[:,:2]),axis=0))
    uv_pts = np.transpose(np.concatenate((bot_eps_uv[:,:2],top_eps_uv[:,:2]),axis=0))
    R,b = rigid_transform_3D(uv_pts,target_pts)
    bot_eps_uv_ = np.transpose(np.matmul(R,np.transpose(bot_eps_uv[:,:2]))+b)
    bot_eps_uv_ = np.concatenate((bot_eps_uv_,bot_eps_uv[:,2:3]),axis=1)
    top_eps_uv_ = np.transpose(np.matmul(R,np.transpose(top_eps_uv[:,:2]))+b)
    top_eps_uv_ = np.concatenate((top_eps_uv_,top_eps_uv[:,2:3]),axis=1)

    if plot:
        fig = plt.figure()
        plt.title('After UV-image rigid alignment')
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.scatter(top_eps_uv_[:,0],top_eps_uv_[:,1],top_eps_uv_[:,2],antialiased=True, alpha=0.05, color='b')
        ax.scatter(top_eps_flat[:,0],top_eps_flat[:,1],top_eps_flat[:,2],antialiased=True, alpha=0.05, color='b')
        ax.scatter(bot_eps_uv_[:,0],bot_eps_uv_[:,1],bot_eps_uv_[:,2],antialiased=True,alpha=0.05,color='r')
        ax.scatter(bot_eps_flat[:,0],bot_eps_flat[:,1],bot_eps_flat[:,2],antialiased=True, alpha=0.05, color='r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(['Top, UV','Top, original','Bot, UV','Bot, original'])
        ax.set_zlim(top_eps_flat[:,2].min()-200, bot_eps_flat[:,2].max()+200) 

    # Return the surface endpoints (potentially filtered), as well as aligned UV maps 
    return top_eps_flat, top_eps_uv_, bot_eps_flat, bot_eps_uv_



def compute_thickness(top_eps, top_eps_uv, bot_eps, thickness_filter=False, nns=10, n_std=2, plot_hist=False):
    '''
    Compute thickness of tissue.
    Potentially filter out points that are far away from mean thickness.
    '''

    kdt = KDTree(bot_eps[:,:2], leaf_size=30) 
    _, inds = kdt.query(top_eps[:,:2], k=nns) # finding which bottom_eps are closest laterally to top_eps


    # Now we can compute the "average" thickness
    thicknesses = []
    for idx in range(len(inds)):
        thickness = np.abs(bot_eps[inds[idx]][:,2].mean() - top_eps[idx,2]) 
        thicknesses.append(thickness)

    thicknesses = np.array(thicknesses)
    print("Mean thickness:",thicknesses.mean())


    # Filter out thicknesses that are n std dev away
    if thickness_filter:
        thickness_mean = thicknesses.mean()
        thickness_std = thicknesses.std()
        low_thresh = thickness_mean - n_std*thickness_std
        high_thresh = thickness_mean + n_std*thickness_std
        print("Low threshold thickness:",low_thresh)
        print("High threshold thickness:",high_thresh)

        top_eps_flat = top_eps[(thicknesses > low_thresh)*(thicknesses < high_thresh)]
        top_eps_uv = top_eps_uv[(thicknesses > low_thresh)*(thicknesses < high_thresh)]
    else:
        top_eps_flat = top_eps.copy()

    # check thicknesses
    if plot_hist:
        plt.figure()
        plt.hist(thicknesses)
        plt.xlabel('Thickness (pixel)')
        plt.ylabel('Frequency')
        plt.show()

    return top_eps_flat, top_eps_uv


#########################

def save_mesh_as_stl(mesh_path, mesh):
    '''
    Save an open3d mesh as STL file for COMSOL
    '''
    
    m = trimesh.Trimesh(np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
        vertex_normals=np.asarray(mesh.vertex_normals))
    m.export(mesh_path)
    return m 

def remove_radius_outliers(pcloud, nb_points, radius):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcloud)
    pcd_filtered,_ = pcd.remove_radius_outlier(nb_points, radius)
    return np.asarray(pcd_filtered.points)

def remove_statistical_outliers_3d(pcloud, num_nn, std_dev_ratio):
    '''
    Utilizes the open3d function instead of custom, used for filtering in 3d instead of by lateral
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcloud)
    pcd_filtered,_ = pcd.remove_statistical_outlier(num_nn, std_dev_ratio)
    return np.asarray(pcd_filtered.points)


def downsample_point_cloud(pcloud, voxel_size):
    '''
    Downsample points in a point cloud. Currently using the Open3D implementation

    voxel_size - the dimension of voxel (in pixels) for which we choose one point 
                (i.e. the number of times to downsample)
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcloud)
    dpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(dpcd.points)

def remove_statistical_outliers(pcloud, num_nn, std_dev_threshold, nn_mode='lateral', dist_mode='z', return_outliers=False):
    '''
    Remove statistical outliers in a point cloud.

    Inputs:
    pcloud - array, point cloud coordinates 
    num_nn - int, number of nearest neighbors to use 
    std_dev_threshold - float, number of std. deviations away tolerated 
    nn_mode - str, options are 'lateral' and 'volumetric'. 
                    if 'lateral', compute nearest neighbors based on xy distance.
                    if 'volumetric', compute nearest neighbors based on full distance.
    dist_mode - str, options are 'z','zslope','volumetric'
                    if 'z', compute distances based on z distance
                    if 'zslope', compute distances based on z slopes (dz / d(xy))
                    if 'volumetric', compute distances based on full distance.  
    '''

    inliers_z = pcloud.copy()

    change = 100; itera = 0
    while change != 0 and itera == 0:
        if nn_mode=='lateral':
            eps_lateral = inliers_z[:,:2]
        else:
            eps_lateral = inliers_z 

        current_inlier_length = len(inliers_z)
        #D = distance.squareform(distance.pdist(eps_lateral))
        #Dinds = np.argsort(D,axis=1)
        #knn_inds = Dinds[:,1:nearest_nbs+1]
        kdt = KDTree(eps_lateral, leaf_size=40) 
        dists, inds = kdt.query(eps_lateral, k=num_nn+1)

        if dist_mode=='z':
            # Compute teh z_dist to nearest neighbors of each 
            inliers_z_ = np.repeat(np.expand_dims(inliers_z[:,2],axis=1),num_nn,axis=1)
            Dz_nns_means = np.mean(np.abs(inliers_z_-inliers_z[inds[:,1:]][:,:,2]),axis=1)
        else:
            # volumetric, so compute total distance to nearest neighbors of each 
            Dz_nns_means = np.mean(dists[:,1:], axis=1)

        #Dz_nns = np.array([Dz[i,knn_inds[i]] for i in range(len(Dz))])
        #Dz_nns_means = np.mean(Dz_nns, axis=1)
        
        if itera==0:
            Dz_nns_means_std = np.std(Dz_nns_means)
            Dz_nns_means_mean = np.mean(Dz_nns_means)

            print('Std. dev:',Dz_nns_means_std)
            print('Mean:',Dz_nns_means_mean)
        
        if return_outliers:
            outliers_z = inliers_z[Dz_nns_means-Dz_nns_means_mean >= std_dev_threshold*Dz_nns_means_std]
        inliers_z = inliers_z[Dz_nns_means-Dz_nns_means_mean < std_dev_threshold*Dz_nns_means_std]
        new_inlier_length = len(inliers_z)

        itera += 1
        change = current_inlier_length - new_inlier_length 
        # print("Number of outliers:",len(outliers_z))
        print("Number of inliers:",len(inliers_z))
    if return_outliers:
        return inliers_z, outliers_z
    else:
        return inliers_z


def gradient_remove_outliers(tcoords, num_nn, threshold=None, plot=False, return_grads=False):
    '''
    Remove surface outliers based on the estimated gradient at each (x,y) point 
    '''
    mags = estimate_pcloud_gradient(tcoords,num_nn)
    if plot:
        plt.figure(figsize=(4,4))
        plt.hist(mags)
        plt.show(block=False)
        plt.pause(0.1)

    if threshold is None:
        threshold = input('Enter threshold to filter with:')
        threshold = float(threshold)

    tcoords_ = tcoords[mags < threshold] # inliers
    tcoords__ = tcoords[mags >= threshold] # outliers 
    if plot:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tcoords_[:,0],tcoords_[:,1],tcoords_[:,2],alpha=0.35)
        ax.scatter(tcoords__[:,0],tcoords__[:,1],tcoords__[:,2],alpha=0.35)
        ax.view_init(azim=0,elev=90)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlim(tcoords_[:,2].max()-200,tcoords_[:,2].max()+200)
        ax.legend(['Inliers','Outliers'])

    if return_grads:
        return tcoords_, mags
    else:
        return tcoords_ 

def kmeans_remove_outliers(pts,num_nn,plot=False):
    '''
    Use Kmeans to remove outliers on the surface
    '''

    mags = estimate_pcloud_gradient(pts,num_nn)

    km = KMeans(n_clusters=2,random_state=42).fit(mags.reshape(-1,1))
    inliers = pts[km.labels_==0]
    outliers = pts[km.labels_==1]

    if plot:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(inliers[:,0],inliers[:,1],inliers[:,2],alpha=0.35)
        ax.scatter(outliers[:,0],outliers[:,1],outliers[:,2],alpha=0.35)
        plt.title('Gradient magnitude')
        plt.show()

    return inliers 

def estimate_pcloud_gradient(pts, num_nn, return_components=False):
    kdt = KDTree(pts[:,:2], leaf_size=40) # the descriptor is still based on 2D 
    _, inds = kdt.query(pts[:,:2], k=num_nn+1)

    mags = []; dxs=[]; dys=[]
    for (i,pt) in tqdm(enumerate(pts)):
        nns = pts[inds[i]]
        C, X, Y, Z = fit_surface(nns, 100, 0, 1, zflip=False, zadd=False, plot=False)
        dx = C[0]; dy = C[1] # estimated gradient
        mag = np.sqrt(dx**2 + dy**2)
        mags.append(mag)
        dxs.append(dx); dys.append(dy)
    mags = np.array(mags) 

    if return_components:
        return mags, np.array(dxs), np.array(dys)
    else:
        return mags  

##########3 Create meshes using open3d ##########
def save_mesh(output_path, mesh):
    o3d.io.write_triangle_mesh(output_path, mesh)

def save_tetra_mesh(output_path, mesh):
    o3d.io.write_tetra_mesh(output_path, mesh)


def create_surface_mesh(point_cloud, faces, output_path=None):
    triangles = o3d.utility.Vector3iVector(faces)
    vertices = o3d.utility.Vector3dVector(point_cloud)
    
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    
    # Prune the mesh 
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    if output_path is not None:
        save_mesh(output_path, mesh)

    return mesh

def uv_map(input_path, output_path, bff_path=bff_global_path):
    cmd2run = bff_path + ' ' + input_path + ' ' + output_path 
    subprocess.call(cmd2run, shell=True)
    

def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods={}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i]=mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods


def pcloud_to_triangles(points, plot=True, alpha=None):
    '''
    Returns a Triangulation object in Matplotlib

    if alpha is None, just do simple Delaunay triangulation
                    else, calculate concave hull 
    '''

    if alpha is None:
        triD = Delaunay(points[:,:2])
        faces = triD.simplices
    else:
        _,_,triangles = alpha_shape(points[:,:2],alpha=alpha,plot=True)
        faces = np.zeros(triangles.shape[:2],dtype='int')
        for i,triangle in tqdm(enumerate(triangles)):
            faces[i] = [np.argwhere((points[:,0]==triangle[0,0])*(points[:,1]==triangle[0,1]))[0][0],
                       np.argwhere((points[:,0]==triangle[1,0])*(points[:,1]==triangle[1,1]))[0][0],
                       np.argwhere((points[:,0]==triangle[2,0])*(points[:,1]==triangle[2,1]))[0][0]]

    if plot:
        plt.figure()
        plt.triplot(points[:,0],points[:,1],faces)
        plt.plot(points[:,0],points[:,1],'o',alpha='0.05')
        plt.show()
    return faces

def pcloud_to_mesh_volumetric(pcloud, output_path=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
    mesh,_ = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    # Prune the mesh 
    mesh.remove_degenerate_tetras()
    mesh.remove_duplicated_tetras()
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_vertices()

    if output_path is not None:
        save_mesh(output_path, mesh)
    return mesh 

def pcloud_to_mesh_poisson(pcloud, output_path=None, **poisson_kwargs):
    '''
    Uses the poisson method to create mesh from point cloud

    poisson_kwargs: (defaults)
    depth=8, 
    width=0, 
    scale=1.1, 
    linear_fit=False

    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
    mesh,_ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, **poisson_kwargs)
    # Prune the mesh 
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    if output_path is not None:
        save_mesh(output_path, mesh)
    return mesh 

def pcloud_to_mesh_ball_pivoting(pcloud, radii, output_path=None):
    '''
    Use the ball pivoting method to create a mesh from point cloud 
    radii - 2-element array containing radii of ball used for algorithm
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

    # Prune the mesh 
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    if output_path is not None:
        save_mesh(output_path, mesh)
    return mesh 


    ############## Extracting UV map from OBJ file ####


def extract_uvmap_from_obj(obj_path, z_coord):
    # Extracts the uvmap vertices that correspond to the correct vertices indicically from mesh file 
    # z_coord is the coordinate which we'd like to have the UV Map to adopt
   
    uv = pd.read_csv(obj_path,delimiter=' ',header=None)

    uvmapped = uv[uv[0]=='vt']
    uvmapped_array = np.asarray(uvmapped)[:,1:3].astype('float')
    uvmappedb_faces = uv[uv[0]=='f']
    uvmappedb_vs = uv[uv[0]=='v']
    uvmapped_arrayb_v = np.array(uvmappedb_vs)[:,1:4].astype('float')
    
    coords_uv_prelim = np.hstack((uvmapped_array[:,:2],z_coord*np.ones((uvmapped_array.shape[0],1))))
    
    uvmappedb_faces_v0 = np.array(uvmappedb_faces[1].apply(lambda x: x.split('/')[0]).astype('int'))-1
    uvmappedb_faces_v1 = np.array( uvmappedb_faces[2].apply(lambda x: x.split('/')[0]).astype('int'))-1
    uvmappedb_faces_v2 = np.array( uvmappedb_faces[3].apply(lambda x: x.split('/')[0]).astype('int'))-1
    uv_bottom_v_inds = np.hstack((uvmappedb_faces_v0,uvmappedb_faces_v1,uvmappedb_faces_v2))
    # print(uv_bottom_v.shape)

    # Texture index coordinates
    uvmappedb_faces_vt0 = np.array( uvmappedb_faces[1].apply(lambda x: x.split('/')[1]).astype('int'))-1
    uvmappedb_faces_vt1 = np.array( uvmappedb_faces[2].apply(lambda x: x.split('/')[1]).astype('int'))-1
    uvmappedb_faces_vt2 = np.array( uvmappedb_faces[3].apply(lambda x: x.split('/')[1]).astype('int'))-1
    uv_bottom_vt_inds = np.hstack((uvmappedb_faces_vt0,uvmappedb_faces_vt1,uvmappedb_faces_vt2))
    # print(uv_bottom_vt.shape)

    vt_inds_dict = []
    for j in range(uv_bottom_v_inds.max()+1):
        vt_inds_dict.append(uv_bottom_vt_inds[np.argwhere(uv_bottom_v_inds == j)[0][0]])

    # print(vt_inds_dict)

    return coords_uv_prelim[vt_inds_dict], uvmapped_arrayb_v 

################### Other geoemtry 

# function for Heron's formula
def calc_triangle_area(coords):
    '''
    Computes area of triangle using Heron's formula 
    '''
    a = np.linalg.norm(coords[0]-coords[1])
    b = np.linalg.norm(coords[0]-coords[2])
    c = np.linalg.norm(coords[1]-coords[2])
    s = (a+b+c)/2
    return np.sqrt(s*(s-a)*(s-b)*(s-c))


################### Filtering out endpoints 

def polygon_remove(pcloud, pts, plane='xy'):
    '''
    Remove points from a polygon defined in a certain plane.
    '''
    if plane=='xy':
        pcloud_ = pcloud[:,:2]
    elif plane=='yz':
        pcloud_ = pcloud[:,1:]
    elif plane=='xz':
        pcloud_ = pcloud[:,np.r_[0,2]]
    
    p = mpl.path.Path(pts)
    flags = p.contains_points(pcloud_)
    pcloud_new = pcloud[~flags]
    return pcloud_new 
    

# def display_pts_projection(pcloud, plane='xy'):
#     '''
#     Plot points in 2D plane 
#     '''
#     global ix, iy
#     global coords_global
#     coords_global = []
#     def onclick(event):
#         ix, iy = event.xdata, event.ydata
#         print ('x = %d, y = %d'%(ix, iy))
        
#         coords_global.append((ix, iy))
#         return coords_global 

#     if plane=='xy':
#         pcloud_ = pcloud[:,:2]
#     elif plane=='yz':
#         pcloud_ = pcloud[:,1:]
#     elif plane=='xz':
#         pcloud_ = pcloud[:,np.r_[0,2]]
        
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(111)
#     ax.scatter(pcloud_[:,0],pcloud_[:,1],alpha=0.35)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     plt.show()
#     cid = fig.canvas.mpl_connect('button_press_event', onclick)
#     return cid, fig 

# def disconnect_display_pts_projection(cid, fig):
#     fig.canvas.mpl_disconnect(cid)
#     pts = np.array(coords_global)
#     return pts 


############ Surface functions ###################

def sample_surface(surf_zarr_path, grid_size=(100,100), save_path=None):
    '''
    Creates a point cloud based on 2D grid_size and a binary surface image
    '''
    if type(surf_zarr_path) is not str:
        thinsurf = surf_zarr_path 
    elif os.path.isdir(surf_zarr_path):
        z = zarr.open(surf_zarr_path,mode='r')
        thinsurf = z[:]
    else:
        thinsurf = io.readData(surf_zarr_path)

    surfcoords = np.argwhere(thinsurf)
    xr,yr = thinsurf.shape[:2]

    xs = np.round(np.linspace(0,xr-1,grid_size[0]))
    ys = np.round(np.linspace(0,yr-1,grid_size[1]))

    X,Y = np.meshgrid(xs,ys)
    coordsxy = np.array([X.flatten(),Y.flatten()]).T.astype('int')

    coordsxy_set = set([tuple(v) for v in coordsxy])
    surfxy_set = set([tuple(v) for v in surfcoords[:,:2]])

    intersectxy = surfxy_set.intersection(coordsxy_set)

    sampled_surf_pts = []
    for coord in tqdm(intersectxy):
        coord_toadd = surfcoords[(surfcoords[:,0]==coord[0]) * (surfcoords[:,1]==coord[1])]
        sampled_surf_pts.append(coord_toadd)
    sampled_surf_pts = np.array(sampled_surf_pts).squeeze()

    if save_path is not None:
        np.save(save_path, sampled_surf_pts)

    return sampled_surf_pts 
    



def alpha_shape(points, alpha, plot=False):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    
    if points.shape[1] > 2:
        points = points[:,:2]
        
    coords = points.copy()
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    
    concave_hull = cascaded_union(triangles)
    if plot:
        plt.figure()
        plt.scatter(coords[:,0],coords[:,1], alpha=0.1)
        plt.plot(*concave_hull.exterior.xy)
        plt.scatter(concave_hull.exterior.xy[0],concave_hull.exterior.xy[1],alpha=0.1,color='r')
        plt.show()
    # find the full pts 
    concave_hull_pts = np.vstack((np.array(concave_hull.exterior.xy[0]),np.array(concave_hull.exterior.xy[1]))).T.astype('int')
    return concave_hull_pts, edge_points, filtered # return the concave_hull, the edges, and the faces 


###############
def filter_manual_surface_points(surf_eps, img_size, downsample_factor=(1,1,1), mesh_path=None, num_pts=10000, alpha=0.01):
    surf_eps = surf_eps.astype('float')
    surf_eps[:,0] *= downsample_factor[0]
    surf_eps[:,1] *= downsample_factor[1]
    surf_eps[:,2] *= downsample_factor[2]

    if mesh_path is not None:
        m = o3d.io.read_triangle_mesh(mesh_path)
        pts = np.asarray(m.sample_points_poisson_disk(num_pts).points)

        pts[:,0] *= downsample_factor[0]
        pts[:,1] *= downsample_factor[1]
        pts[:,2] *= downsample_factor[2]
    else:
        pts = surf_eps 

    img_size = [int(img_size[j]*downsample_factor[j]) for j in range(3)]
    grid_x, grid_y = np.mgrid[0:img_size[0],0:img_size[1]]
    points = pts[:,:2]
    values = pts[:,2]
    grid_z_outside = griddata(points,values,(grid_x,grid_y),method='nearest') # to account for inability to interpolate outside convex hull 
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')
    grid_z0[np.isnan(grid_z0)] = grid_z_outside[np.isnan(grid_z0)]
    grid_z0 = grid_z0.astype('int')
    surf_pts = np.vstack((np.argwhere(grid_z0)[:,0],np.argwhere(grid_z0)[:,1],grid_z0[(np.argwhere(grid_z0)[:,0],np.argwhere(grid_z0)[:,1])])).T

    surf_array = np.zeros(img_size,dtype='uint8')
    surf_array[(surf_pts[:,0],surf_pts[:,1],surf_pts[:,2])] = 1

    concave_hull, edges, _ = alpha_shape(surf_eps, alpha, plot=True)
    img = Image.new('L', img_size[:2], 0)
    ImageDraw.Draw(img).polygon([tuple(border_pt.astype('int')) for border_pt in concave_hull[:,:2]], outline=1, fill=1)
    mask = np.array(img).T

    filtered_img = np.stack([surf_array[:,:,k] * mask for k in range(surf_array.shape[2])], axis=2)

    return filtered_img, surf_pts  