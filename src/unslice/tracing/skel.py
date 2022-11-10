import numpy as np
import cupy as cp 
from . import sknw 
from tqdm import tqdm 
import zarr
import time 
from .. import IO as io 
from ..utils import get_chunk_coords, numpy_to_json
from ..registration.rigid import define_rigid_transform, rigid_transform 
import networkx as nx
import multiprocessing as mp 
from functools import partial  
from skimage.morphology import skeletonize_3d 
from skimage.morphology import skeletonize as skeletonize_2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numcodecs import Blosc,LZ4,Zstd

# Tools for skeletonizing the image 
DISPLAY_SKEL = False 

def skeletonize(img):
    '''
    Skeletonize a binary image (CPU support).
    
    Inputs:
    img - nD array to be skeletonized 
    
    Outputs:
    skeletonized nD array
    '''
    
    if len(img.shape) == 2:
        return skeletonize_2d(img)
    elif len(img.shape) == 3:
        return skeletonize_3d(img) 
    else:
        raise RuntimeError('Image dimension needs to be 2 or 3, found %d instead'%(len(img.shape)))
        
def calculate_network_length(graph):
    '''
    Calculate the total length of a networkx graph, as the number of nodes and links.
    '''

    # Other possibility is to use the edge weights (which are the lengths)
    
    # num_nodes = len(list(graph.nodes))
    num_nodes = 0 # some nodes have more than one pixel
    for node in list(graph.nodes):
        num_nodes += len(graph.nodes[node]['pts'])
    num_links = 0
    for edge in list(graph.edges):
        num_links += len(graph[edge[0]][edge[1]]['pts'])
    return num_links + num_nodes 

def _remove_small_groups(gr, min_cc):
    '''
    Get rid of small groups of voxels with connected components less than min_cc

    Inputs:
    gr - networkx graph
    min_cc - minimum number of connected components
    '''

    cs = nx.connected_components(gr)
    cslist = [c for c in sorted(cs,key=len,reverse=True)]
    cslistlens = [len(c) for  c in cslist]
    c = np.array(cslist)
    # Check all components with 3 nodes or less; it is unlikely we remove clusters with more nodes 
    cc_candidates = list(c[np.array(cslistlens) < 4]) 

    edges = np.array(list(gr.edges))
    for nodeset in cc_candidates:
        nodes = [s for s in nodeset]
        inds = np.array([])
        for node in nodes:
            inds = np.append(inds,np.argwhere(edges == node)[:,0])
        inds = np.unique(inds).astype('int')
        
        num_ccs = len(nodes)
        for ind in inds:
            num_ccs += len(gr[edges[ind][0]][edges[ind][1]]['pts'])
        if num_ccs < min_cc:
            for node in nodes:
                gr.remove_node(node)

    return gr


def get_ep_direction(graph, num_points=5):
    '''
    Calculates the directionality vector of all endpoints in a graph

    Inputs:
    graph - networkx graph, with nodes that are endpoints having attribute "is_ep"==True
    num_points - the number of points in the branch to factor in for calculating directionality

    Returns:
    dictionary with endpoint index as keys and phi,theta (in degrees) as values
    '''

    ep_dict = dict(graph.nodes(data='is_ep'))
    ep_inds = [k for k,v in ep_dict.items() if v == True]

    angle_avgs = [] 
    for i in ep_inds: 
        nb_node = list(graph.neighbors(i))[0]
        nb_pt = graph.node[nb_node]['o']
        pt = graph.node[i]['o']
        branch_pts = graph[nb_node][i]['pts']

        # flip the branch pts if the endpt in question is at the other end 
        if np.linalg.norm(pt-branch_pts[0]) > np.linalg.norm(pt-branch_pts[-1]):
            branch_pts = np.flip(branch_pts,axis=0)

        if num_points is not None:
            branch_pts_true = branch_pts[:num_points]
        else:
            branch_pts_true = branch_pts

        # Averages the direct vector angles from each branch point to endpoint 
        angles = []
        for branch_pt in branch_pts:
            vec = pt-branch_pt
            r = np.sqrt(np.sum(np.square(vec)))
            phi = np.arctan2(vec[1],vec[0])#*180/np.pi
            theta = np.arccos(vec[2]/r)#*180/np.pi
            angles.append([r,phi,theta])
        angles = np.array(angles)

        # Compute the average angle, in radians 
        theta_avg = np.mean(angles[:,2]) # decline angle, measured from positive z axis
        phi_avg = np.mean(angles[:,1]) # azimuthal, measured from positive x axis
        angle_avgs.append([phi_avg,theta_avg])

    angle_avgs = np.array(angle_avgs)*180/np.pi
    return dict(zip(ep_inds, angle_avgs))


def _remove_clustered_eps(endpoints, radii, orientation):
    '''
    Searches through the endpoints to get rid of clusters of endpoints from same vessel.
    Keeps the endpoints that are the topmost (or bottommost) in z within an area defined by radii.
    
    Inputs:
    endpoints - numpy ndarray, Nx3 matrix of endpoints
    radii - tuple (x,y) the x and y radii around which to filter endpoints
    orientation - 'top' or 'bottom', 'top' means the surface is closer to z=0, 
                  'bottom' means the surface is closer to z=end
    '''
    xr,yr = radii  

    change = 100
    while change > 0: # stop the loop when the number of endpoints stop changing 
        current_length = endpoints.shape[0]
        surf_eps = []
        for ep in endpoints:
            x,y,z = ep
            candidates = endpoints[(endpoints[:,0]>=x-xr) * (endpoints[:,0]<x+xr) *\
                                    (endpoints[:,1]>=y-yr) * (endpoints[:,1]<y+yr)]
            if orientation == 'top':
                to_keep = candidates[np.argmin(candidates[:,2])]
            elif orientation == 'bottom':
                to_keep = candidates[np.argmax(candidates[:,2])]
            surf_eps.append(to_keep)  
        endpoints_new = np.unique(np.array(surf_eps),axis=0)
        change = endpoints.shape[0] - endpoints_new.shape[0]
        endpoints = endpoints_new

    return endpoints_new 



def skel2graph3d(skel, min_branch_length=2, min_cc=2):
    '''
    Turns skeleton into a networkx graph 
    
    Inputs:
    skel - nD binary array containing skeleton 
    min_branch_length - int, smallest branch length allowable (default: 2, i.e. remove all 1-length branches)
    min_cc- int, smallest number of voxels in a connected component (to remove spurious detections)
    
    Outputs:
    graph - filtered networkx graph 
    '''

    graph = sknw.build_sknw(skel)
    graph.remove_edges_from(graph.selfloop_edges()) # removes self loop edges 
    graph_copy = graph.copy()

    # Clean up the graph     
    for i in graph_copy.nodes():
        if i in graph.nodes(): # since graph_copy never gets removed
            key_nodes = list(graph.neighbors(i)) 
            if len(key_nodes) == 0:
                # Remove all isolated nodes
                graph.remove_node(i)
            elif len(key_nodes) == 1: 
                graph.node[i]['is_ep'] = True 
            else: 
                for key_node in key_nodes:
                    if len(list(graph.neighbors(key_node))) == 1:
                        # only remove the other node if it's an endpoint 
                        if len(graph[key_node][i]['pts']) < min_branch_length - 1:
                            graph.remove_node(key_node)
                # After potentially removing the other node, we check again for this node: 
                if len(list(graph.neighbors(i))) == 0:
                    graph.remove_node(i)
                elif len(list(graph.neighbors(i))) == 1:
                    graph.node[i]['is_ep'] = True
                else:
                    graph.node[i]['is_ep'] = False 

    # remove small clusters of endpoints 
    graph = _remove_small_groups(graph, min_cc)


    
    return graph

  
def graph2skel3d(graph, img_shape):
    '''
    Turn the graph back into a skeleton. 
    
    Inputs:
    graph - networkx graph 
    img_shape - tuple containing the shape of the image 
    
    Outputs:
    skel - nD binary array skeleton 
    '''
    
    skel = np.zeros(img_shape, dtype='uint8')
    
    if len(list(graph.nodes)) > 0:
        # Get all of the coordinates in the graph 
        coords = np.zeros((0,3),dtype='uint16')
        for i in list(graph.nodes):
            if len(list(graph.edges(i))) > 0:
                coords = np.concatenate((coords,graph.nodes[i]['pts']),axis=0)
                
        # Now add all the edge coordinates that exist 
            for j in list(graph.neighbors(i)):
                coords = np.concatenate((coords,graph[j][i]['pts']),axis=0)
        
        coords_final = np.unique(coords,axis=0)
        skel[tuple(coords_final.T)] = 1
    
    return skel
    
def get_endpoints(graph, surface=None, prune_directionality_num_points=None, orientation='top', z_shape=None):
    '''
    Takes a Networkx graph and returns the endpoints.


    Inputs:
    graph - Networkx graph 
    surface - if not None, nD binary array containing where we should look for endpoints 
    prune_directionality_num_points - if not None, then prune the endpoints based on directionality calculated using num_points
    orientation - 'top' or 'bottom': if 'top', the surface is closer to z=0. if 'bottom', the surface is closer to z=end.
    z_shape - int, the z shape of the image. default None, only matters if orientation=='bottom'

    Outputs:
    endpoints - N*3 array containing the endpoint coordinates 
    '''
    
    endpoints = []
    nodes = graph.node
    # Remove endpoints that are trending away from the surface
    if prune_directionality_num_points is not None:
        ep_directions = get_ep_direction(graph, num_points=prune_directionality_num_points)
        for i in list(ep_directions.keys()): # These are the node indices 
            if ep_directions[i][1] > 90.0: # incline angle is facing "towards z=0"
                if orientation == 'top':
                    endpoints.append(np.round(nodes[i]['o']).astype('uint16'))
            elif orientation == 'bottom':
                endpoints.append(np.round(nodes[i]['o']).astype('uint16'))
    else:
        # Construct array containing all the endpoint coordinates 
        for i in nodes:
            if nodes[i]['is_ep']:
                endpoints.append(np.round(nodes[i]['o']).astype('uint16'))

    endpoints = np.asarray(endpoints) 

    if surface is not None and len(endpoints) > 0:
        endpoint_surf = np.zeros(surface.shape,dtype='uint8')
        # surface might not be same shape as graph --> only get the ones on surface
        if orientation == 'top':
            endpoints = endpoints[(endpoints[:,2] < surface.shape[2])] 
        elif orientation == 'bottom':
            endpoints = endpoints[(endpoints[:,2] >= z_shape-surface.shape[2])]
        endpoint_surf[(endpoints[:,0],endpoints[:,1],endpoints[:,2])] = 1 
        endpoints = np.argwhere(surface * endpoint_surf)

    return endpoints 
            
   
############# Main functions - putting it all together ############   
def trace_zarr(source_zarr_path, surface_zarr_path, sink_zarr_path, min_branch_length, 
               min_cc, overlap=0, num_workers=8, chunks=None, apply_oof=False, sample_coord_ranges=None, **kwargs):
    '''
    Skeletonize, turn into graph, and detect endpoints after tracing 
    
    Inputs:
    source_zarr_path - 
    surface_zarr_path - 
    sink_zarr_path -
    min_branch_length - take out branches of this length or smaller 
    min_cc - minimum allowed connected components to stay in image  
    overlap - number of pixels overlap (default: 0)
    num_workers - (default: 8) 
    chunks - (default: source_zarr.chunks)
    apply_oof - bool, if True, apply OOF before tracing 
    sample_coord_ranges - array of arrays, if not None, process just the small coord ranges

    Outputs:
    eps - endpoints 
    '''

    prune_directionality_kwargs = {}
    remove_clustered_eps_kwargs = {}
    if 'prune_directionality_num_points' in kwargs:
        prune_directionality_kwargs['prune_directionality_num_points'] = kwargs['prune_directionality_num_points']
    if 'orientation' in kwargs:
        prune_directionality_kwargs['orientation'] = kwargs['orientation']
    if 'z_shape' in kwargs:
        prune_directionality_kwargs['z_shape'] = kwargs['z_shape']
    if 'radii' in kwargs:
        remove_clustered_eps_kwargs['radii'] = kwargs['radii'] 
        if 'orientation' in kwargs:
            remove_clustered_eps_kwargs['orientation'] = kwargs['orientation']

    source = zarr.open(source_zarr_path, mode='r')
    if surface_zarr_path is not None:
        surface = zarr.open(surface_zarr_path, mode='r') 
    else:
        surface = None 

    if chunks is None:
        chunks = source.chunks 
    #sink = zarr.zeros(shape=source.shape, chunks=chunks, dtype=source.dtype, 
    #                  store=zarr.DirectoryStore(sink_zarr_path), overwrite=True)

    # We try with a different compressor to see if it helps the random issues 
    sink = zarr.zeros(shape=source.shape, chunks=chunks, dtype=source.dtype,
                      store=zarr.DirectoryStore(sink_zarr_path), overwrite=True)#,
                      #compressor=Zstd(level=1))

    if sample_coord_ranges is not None:
        for sample_coord_range in sample_coord_ranges:
            xrs,yrs,zrs = sample_coord_range
            sample_coord_range_test = np.array(get_chunk_coords((xrs[1]-xrs[0],yrs[1]-yrs[0],zrs[1]-zrs[0]), sink.chunks))
            sample_coord_range_test[:,0,:] += xrs[0] 
            sample_coord_range_test[:,1,:] += yrs[0]
            sample_coord_range_test[:,2,:] += zrs[0] 

            f = partial(trace_chunk, source, surface, sink, min_branch_length, min_cc, overlap, **prune_directionality_kwargs ) 
            p = mp.Pool(num_workers)
            eps = list(tqdm(p.imap(f, sample_coord_range_test), total=len(sample_coord_range_test)))
            p.close()
            p.join() 

            eps = np.vstack(tuple(eps)) 
            if len(remove_clustered_eps_kwargs) > 0:
                eps = _remove_clustered_eps(eps, **remove_clustered_eps_kwargs)
            eps[:,0] -= xrs[0]; eps[:,1] -= yrs[0]; eps[:,2] -= zrs[0]

            sample_source_path = sink_zarr_path[:-5] +'_x%d_%d_y%d_%d_z%d_%d'%\
                                (xrs[0],xrs[1],yrs[0],yrs[1],zrs[0],zrs[1]) + '_original.tif'
            sample_sink_path = sink_zarr_path[:-5] + '_x%d_%d_y%d_%d_z%d_%d'%\
                                (xrs[0],xrs[1],yrs[0],yrs[1],zrs[0],zrs[1]) + '_skel.tif'
            eps_path = sample_sink_path[:-9]+'_eps.json'
            #print(eps_path)

            original = source[xrs[0]:xrs[1],yrs[0]:yrs[1],zrs[0]:zrs[1]]
            segged = sink[xrs[0]:xrs[1],yrs[0]:yrs[1],zrs[0]:zrs[1]]

            io.writeData(sample_source_path, original)
            io.writeData(sample_sink_path, segged)
            numpy_to_json(eps, eps_path)
            nuggt_cmd = 'nuggt-display '+sample_source_path+' original green '+sample_sink_path+' skel blue --points ' + eps_path +' --ip-address 10.93.6.101 --port=8902'
            print('Display in Nuggt by doing !{cmd}: \n%s'%(nuggt_cmd))
        return 

    else:
        coord_ranges = get_chunk_coords(sink.shape, chunks)  

        p = mp.Pool(num_workers) 
        f = partial(trace_chunk, source, surface, sink, min_branch_length, min_cc, overlap, **prune_directionality_kwargs)
        chunksize = sink.shape[0]//sink.chunks[0] + 1 # each worker is given each row of chunks 
        eps = list(tqdm(p.imap(f, coord_ranges), total=len(coord_ranges)))
        p.close()
        p.join()

        eps = np.vstack(tuple(eps)) 
        if len(remove_clustered_eps_kwargs) > 0:
            eps = _remove_clustered_eps(eps, **remove_clustered_eps_kwargs)
        return eps 
    
    
    
def trace_chunk(source_zarr, surface_zarr, sink_zarr, min_branch_length, min_cc, overlap, coord_range, **prune_directionality_kwargs):
    # surface_zarr - the zarr file of the surface (so that we can identify surface endpoints) 
    # Overlap - the number of pixels in each direction to overlap so we don't have to 
    #           re-establish endpoints 
    # coord_range - list of list of lists of all of the chunks in the zarr. 
    
    start = time.time()
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
        #print(xr,yr,zr)
        img = source_zarr[xr[0]-overlaps[0][0]:xr[1]+overlaps[0][1],
                          yr[0]-overlaps[1][0]:yr[1]+overlaps[1][1],
                          zr[0]-overlaps[2][0]:zr[1]+overlaps[2][1]] 
    else:
        img = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
   # print("Segmented chunk has: %d pixels"%(np.sum(img > 0)))

    skel, graph = skeletonize_graph(img, min_branch_length, min_cc)

    sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]] = \
             skel[overlaps[0][0]:overlaps[0][0]+xr[1]-xr[0],
                  overlaps[1][0]:overlaps[1][0]+yr[1]-yr[0],
                  overlaps[2][0]:overlaps[2][0]+zr[1]-zr[0]]
    #print("Number of pixels",skel.sum()) # just to see if there is the 

    if surface_zarr is not None:
        surface = surface_zarr[xr[0]-overlaps[0][0]:xr[1]+overlaps[0][1],
                              yr[0]-overlaps[1][0]:yr[1]+overlaps[1][1],
                              zr[0]-overlaps[2][0]:zr[1]+overlaps[2][1]] 
    else:
        surface = None 
                          
    eps = get_endpoints(graph, surface=surface, **prune_directionality_kwargs)
                        # prune_directionality_num_points=prune_directionality_num_points,
                        # orientation=orientation, 
                        # z_shape=z_shape)                       
    if len(eps) > 0:
        eps = eps[(eps[:,0]>=overlaps[0][0])*(eps[:,0]<overlaps[0][0]+xr[1]-xr[0])*\
                  (eps[:,1]>=overlaps[1][0])*(eps[:,1]<overlaps[1][0]+yr[1]-yr[0])*\
                  (eps[:,2]>=overlaps[2][0])*(eps[:,2]<overlaps[2][0]+zr[1]-zr[0])]
        if len(eps) > 0:          
            # Now need to transpose these into the global coordinates 
            eps[:,0] += xr[0] - overlaps[0][0] 
            eps[:,1] += yr[0] - overlaps[1][0]
            eps[:,2] += zr[0] - overlaps[2][0] 
        else:
            # If there's no endpoints in this section 
            # print("No endpoints found in chunk x: %d-%d, y:%d-%d,d z: %d-%d"%\
            #         (xr[0],xr[1],yr[0],yr[1],zr[0],zr[1]))
            eps = np.zeros((0,3), dtype='uint16')
    else:
        # If there's no endpoints in this section 
        # print("No endpoints found in chunk x: %d-%d, y:%d-%d,d z: %d-%d"%\
        #         (xr[0],xr[1],yr[0],yr[1],zr[0],zr[1]))
        eps = np.zeros((0,3), dtype='uint16')
    
    # if len(eps) > 0:
        # print("%d endpoints found in chunk x: %d-%d, y:%d-%d,d z: %d-%d"%(len(eps),
        #     xr[0],xr[1],yr[0],yr[1],zr[0],zr[1]))
    # print("Done with chunk x: %d-%d, y:%d-%d,d z: %d-%d in  %f seconds"%\
    #         (xr[0],xr[1],yr[0],yr[1],zr[0],zr[1],time.time()-start))
    return eps 
    
    
def skeletonize_graph(img, min_branch_length, min_cc):
    '''
    Skeletonize and convert into graph

    Inputs:
    img - numpy array, binary image to be skeletonized
    min_branch_length - minimum length of branches
    min_cc - minimum number of voxels in a connected component  
    '''

    skel = skeletonize(img)
    gr = skel2graph3d(skel, min_branch_length, min_cc)
    network_length = calculate_network_length(gr)

    if network_length > 0:
        skel2 = graph2skel3d(gr, img.shape)
        gr2 = skel2graph3d(skel2, min_branch_length, min_cc)
        network_length_new = calculate_network_length(gr2)
    #     print("num nodes is %d"%len(list(gr2.nodes)))
    #     print("network length is %d"%network_length_new)

        # keep updating the network as long as the network has changed more than 0.5%
        if network_length_new > 0: 
            iter = 0 
            while network_length_new > 0 and np.abs(network_length - network_length_new) / network_length > 0.005:
                network_length = network_length_new
                skel2 = graph2skel3d(gr2, img.shape)
                gr2 = skel2graph3d(skel2, min_branch_length)
                network_length_new = calculate_network_length(gr2)
        #         print("num nodes is %d"%len(list(gr2.nodes)))
        #         print('Iteration %d, network length is %d'%(iter,network_length_new))
                iter += 1

        skel2 = graph2skel3d(gr2, img.shape)
    else:
        gr2 = gr
        skel2 = skel 

    return skel2, gr2


def get_mask_endpoints(eps, mask, mask_downsample_factor=None, **remove_clustered_eps_kwargs):
    '''
    Function for filtering a list of endpoints based on if they're in a mask provided.

    Inputs:
    z0 - int, the z slice at which the mask starts 
    '''

    # Transform all the coordinates to be downsampled
    if mask_downsample_factor is not None:
        dx,dy,dz = mask_downsample_factor 
        R,b = define_rigid_transform(Scx=dx,Scy=dy,Scz=dz)
        eps_ds = rigid_transform(R,b,eps)
        eps_ds = np.round(eps_ds)
        eps_ds = eps_ds.astype('int')
    else:
        eps_ds = eps 

    # Get rid of all the endpoints that don't fit into the mask shape
    eps = eps[(eps_ds[:,0]<mask.shape[0]) * (eps_ds[:,1]<mask.shape[1]) * (eps_ds[:,2]<z1) * (eps_ds[:,2]>=z0)]
    eps_ds = eps_ds[(eps_ds[:,0]<mask.shape[0]) * (eps_ds[:,1]<mask.shape[1]) * (eps_ds[:,2]<mask.shape[2])]

    # get the surface endpoints in the image 
    eps_img = np.zeros(mask.shape, dtype='uint8')
    eps_img[(eps_ds[:,0],eps_ds[:,1],eps_ds[:,2])] = 1
    eps_surf = np.argwhere(eps_img * mask)

    # Now get the indices 
    eps_surf_inds = []
    for ep_surf in eps_surf:
        ind = np.argwhere((eps_ds[:,0]==ep_surf[0]) * (eps_ds[:,1]==ep_surf[1]) * (eps_ds[:,2]==ep_surf[2]))[0][0]
        eps_surf_inds.append(ind)

    eps_surf = eps[eps_surf_inds]
    if len(remove_clustered_eps_kwargs) > 0:
        eps_surf = _remove_clustered_eps(eps_surf, **remove_clustered_eps_kwargs)
    return eps_surf 


def _get_mask_endpoints_pkernel(eps, mask_zarr, coord_range):
    '''
    Kernel for parallel version of get_mask_endpoints

    Inputs:
    eps - array of endpoints, could already be downsampled
    z0 - the number of z slices to add to this 

    Returns:
    ep_inds - indices of the endpoints 
    '''

    xr,yr,zr = coord_range 
    mask = mask_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]

    eps_ds = eps.copy() 

    eps_ds[:,0] -= xr[0]
    eps_ds[:,1] -= yr[0]
    eps_ds[:,2] -= zr[0]

    eps_chunk = eps_ds[(eps_ds[:,0]>=0)*(eps_ds[:,0]<mask.shape[0])*\
                       (eps_ds[:,1]>=0)*(eps_ds[:,1]<mask.shape[1])*\
                       (eps_ds[:,2]>=0)*(eps_ds[:,2]<mask.shape[2])]

    eps_img = np.zeros(mask.shape, dtype='uint8')
    eps_img[(eps_chunk[:,0],eps_chunk[:,1],eps_chunk[:,2])] = 1
    eps_surf = np.argwhere(eps_img * mask)

    # Convert back to global coordinates 
    eps_surf[:,0] += xr[0]
    eps_surf[:,1] += yr[0]
    eps_surf[:,2] += zr[0]

    # Now get the indices 
    eps_surf_inds = []
    for ep_surf in eps_surf:
        ind = np.argwhere((eps[:,0]==ep_surf[0]) * (eps[:,1]==ep_surf[1]) * (eps[:,2]==ep_surf[2]))[0][0]
        eps_surf_inds.append(ind)

    return eps_surf_inds 


def get_mask_endpoints_zarr(eps, mask_zarr_path, mask_downsample_factor=None, z0=0, chunks=None, num_workers=8, **remove_clustered_eps_kwargs):
    '''
    Parallel zarr version for get_mask_endpoints

    Inputs:
    chunks - tuple, how to chunk parallelize the function. if None, then use the mask_zarr's chunks 
    z0 - int, the number of z slices to add on to the mask (so it'd be in the mask resolution)
    '''

    if mask_downsample_factor is not None:
        dx,dy,dz = mask_downsample_factor 
        R,b = define_rigid_transform(Scx=dx,Scy=dy,Scz=dz)
        eps_ds = rigid_transform(R,b,eps)
        eps_ds = np.round(eps_ds)
        eps_ds = eps_ds.astype('int')
    else:
        eps_ds = eps 

    eps_ds[:,2] -= z0 # subtract the z0 slice
    eps_ds = eps_ds[eps_ds[:,2]>=0]

    mask = zarr.open(mask_zarr_path, mode='r')

    if chunks is None:
        chunks = mask.chunks 
    coord_ranges = get_chunk_coords(mask.shape, chunks)  

    p = mp.Pool(num_workers) 
    f = partial(_get_mask_endpoints_pkernel, eps_ds, mask)
    eps_inds = list(tqdm(p.imap(f, coord_ranges), total=len(coord_ranges)))
    p.close()
    p.join()
    
    # eps_inds = np.vstack(tuple(eps_inds)) 
    eps_inds= np.concatenate(tuple(eps_inds),axis=0).astype('int')

    eps_surf = eps[eps_inds]
    if len(remove_clustered_eps_kwargs) > 0:
        eps_surf = _remove_clustered_eps(eps_surf, **remove_clustered_eps_kwargs)
    return eps_surf

############## Visualization tools for graph and skeleton #########################

def visualize_graph(gr2, cmap='inferno'):
    '''
    Visualize a skeletonized and associated network graph. Plot the nodes as different colors than the edges 
    '''
    i = 20
    node_component = np.zeros((0,4),dtype='float')
    for edge in list(gr2.edges):
        if gr2.nodes[edge[0]]['is_ep']:
            node_component = np.vstack((node_component, np.concatenate((gr2.nodes[edge[0]]['o'],np.array([0])),axis=0)))
        else:
            node_component = np.vstack((node_component, np.concatenate((gr2.nodes[edge[0]]['o'],np.array([10])),axis=0)))
        if gr2.nodes[edge[1]]['is_ep']:
            node_component = np.vstack((node_component, np.concatenate((gr2.nodes[edge[1]]['o'],np.array([0])),axis=0)))
        else:
            node_component = np.vstack((node_component, np.concatenate((gr2.nodes[edge[1]]['o'],np.array([10])),axis=0)))
        temp = np.concatenate((gr2[edge[0]][edge[1]]['pts'], i*np.ones((len(gr2[edge[0]][edge[1]]['pts']),1))), axis=1)
        node_component = np.concatenate((node_component,temp),axis=0)
        i += 1
    
    print('%d different edges and %d different nodes'%(len(list(gr2.edges)), len(list(gr2.nodes))))

    node_component = np.unique(node_component, axis=0)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    cmhot = plt.get_cmap(cmap)
    ax.scatter(node_component[:,0], node_component[:,1], node_component[:,2], 
               c=node_component[:,3], s=50, cmap=cmhot)


def visualize_skel(skel, eps):
    sk_coords = np.argwhere(skel)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sk_coords[:,0], sk_coords[:,1], sk_coords[:,2], 
               c='blue', s=50, alpha=0.5)
    ax.scatter(eps[:,0], eps[:,1], eps[:,2], 
               c='red', s=100, alpha=1.0)
