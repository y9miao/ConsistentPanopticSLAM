# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

import open3d as o3d
import numpy as np
import trimesh
import open3d as o3d

def eval_mesh(file_pred, file_trgt, threshold=.05, down_sample=.02):
    """ Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points

    Returns:
        Dict of mesh metrics
    """

    pcd_pred = o3d.io.read_point_cloud(file_pred)
    pcd_trgt = o3d.io.read_point_cloud(file_trgt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist1 = nn_correspondance(verts_pred, verts_trgt)
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist1<threshold).astype('float'))
    recal = np.mean((dist2<threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {'completeness': float(np.mean(dist1)),
               'accuracy': float(np.mean(dist2)),
               'prec': float(precision),
               'recal': float(recal),
               'fscore': float(fscore),
               }
    return metrics


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1
    
    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])
    
    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def project_to_mesh(from_mesh_f, to_gt_f, to_mesh_f, dist_thresh=1):
    """ Transfers attributs from from_mesh to to_mesh using nearest neighbors

    Each vertex in to_mesh gets assigned the attribute of the nearest
    vertex in from mesh. Used for semantic evaluation.

    Args:
        from_mesh: Trimesh with known attributes
        to_mesh: Trimesh to be labeled
        attribute: Which attribute to transfer
        dist_thresh: Do not transfer attributes beyond this distance
            (None transfers regardless of distacne between from and to vertices)

    Returns:
        Trimesh containing transfered attribute
    """
    from_mesh = trimesh.load(from_mesh_f)
    to_gt_mesh = trimesh.load(to_gt_f)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(from_mesh.vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    pred_colors = from_mesh.visual.vertex_colors

    matched_colors = np.zeros((to_gt_mesh.vertices.shape[0], 4), dtype=np.uint8)

    for i, vert in enumerate(to_gt_mesh.vertices):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        if dist_thresh is None or dist[0]<dist_thresh:
            matched_colors[i] = pred_colors[inds[0]]

    mesh = to_gt_mesh.copy()
    mesh.visual.vertex_colors = matched_colors
    mesh.export(to_mesh_f)

def get_vertices(ply_data):
	x = ply_data['vertex'].data['x'].reshape(-1,1)
	y = ply_data['vertex'].data['y'].reshape(-1,1)
	z = ply_data['vertex'].data['z'].reshape(-1,1)
	return np.concatenate((x,y,z),axis=1)

def get_colors(ply_data):
	r = ply_data['vertex'].data['red'].reshape(-1,1)
	g = ply_data['vertex'].data['green'].reshape(-1,1)
	b = ply_data['vertex'].data['blue'].reshape(-1,1)
	return np.concatenate((r,g,b),axis=1)

def project_to_mesh(from_mesh_f, to_gt_f, to_mesh_f, dist_thresh=0.05):
    import open3d.core as o3c
    from_mesh = o3d.io.read_point_cloud(from_mesh_f)
    to_gt_mesh = o3d.io.read_point_cloud(to_gt_f)
    from_mesh_vertex = np.asarray(from_mesh.points).astype(np.float32)
    from_mesh_color = np.asarray(from_mesh.colors).astype(np.float32)
    to_gt_mesh_vertex = np.asarray(to_gt_mesh.points).astype(np.float32)

    knn_data = o3c.nns.NearestNeighborSearch(from_mesh_vertex)
    knn_data.knn_index()
    indices, distances_sqr = knn_data.knn_search(to_gt_mesh_vertex, 1)
    indices = indices.reshape(-1)
    matched = (distances_sqr<dist_thresh*dist_thresh).reshape(-1)

    output_color = np.ones((to_gt_mesh_vertex.shape[0], 3)).astype(np.float32) * 200.0/255
    output_color[matched] = from_mesh_color[indices[matched]]
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(to_gt_mesh_vertex)
    out_pcd.colors = o3d.utility.Vector3dVector(output_color)
    o3d.io.write_triangle_mesh(to_mesh_f, out_pcd)

def project_to_mesh_cuda(from_mesh_f, to_gt_f, to_mesh_f, device, dist_thresh=0.05):
    import open3d.core as o3c
    point_dtype = o3c.float32
    color_dtype = o3c.float32

    from_mesh = o3d.io.read_point_cloud(from_mesh_f)
    to_gt_mesh = o3d.io.read_point_cloud(to_gt_f)

    from_mesh_vertex = np.asarray(from_mesh.points).astype(np.float32)
    from_mesh_color = np.asarray(from_mesh.colors).astype(np.float32)
    to_gt_mesh_vertex = np.asarray(to_gt_mesh.points).astype(np.float32)

    from_mesh_vertex_tensor = o3c.Tensor(from_mesh_vertex, dtype=point_dtype, device=device)
    from_mesh_color_tensor = o3c.Tensor(from_mesh_color, dtype=color_dtype, device=device)
    to_gt_mesh_vertex_tensor = o3c.Tensor(to_gt_mesh_vertex, dtype=point_dtype, device=device)

    knn_data = o3c.nns.NearestNeighborSearch(from_mesh_vertex_tensor)
    knn_data.knn_index()

    indices, distances_sqr = knn_data.knn_search(to_gt_mesh_vertex_tensor, 1)
    indices = indices.reshape(-1)
    matched = (distances_sqr<dist_thresh*dist_thresh).reshape(-1)

    output_color = np.ones((to_gt_mesh_vertex.shape[0], 3)).astype(np.float32) * 200.0/255
    output_color_tensor = o3c.Tensor(output_color, dtype=color_dtype, device=device)
    output_color_tensor[matched] = from_mesh_color_tensor[indices[matched]]
    output_color = output_color_tensor.cpu().numpy()

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(to_gt_mesh_vertex)
    out_pcd.colors = o3d.utility.Vector3dVector(output_color)
    o3d.io.write_point_cloud(to_mesh_f, out_pcd)


def eval_depth(depth_pred, depth_trgt):
    """ Computes 2d metrics between two depth maps
    
    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth

    Returns:
        Dict of metrics
    """
    mask1 = depth_pred>0 # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt<10) * (depth_trgt>0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred-depth_trgt)
    abs_rel = abs_diff/depth_trgt
    sq_diff = abs_diff**2
    sq_rel = sq_diff/depth_trgt
    sq_log_diff = (np.log(depth_pred)-np.log(depth_trgt))**2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25**2).astype('float')
    r3 = (thresh < 1.25**3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics