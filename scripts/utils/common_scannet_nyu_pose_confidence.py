
import sys
import os
import time
import threading
import h5py
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import cv2
from collections import Counter

from semantics.pano_scannet_nyu_colormap import *
from multiprocessing import Process

import sys
sys.path.append("/usr/lib/python3/dist-packages")
import pcl

def preprocess(depth_img):
    depth_img_rescaled = None
    if depth_img.dtype == np.uint16:
        # convert depth image from mili-meters to meters
        depth_img_rescaled = cv2.rgbd.rescaleDepth(depth_img, cv2.CV_32FC1)
    elif depth_img.dtype == np.float32:
        depth_img_rescaled = depth_img
    else:
        print("Unknown depth image encoding.")
        return None

    kZeroValue = 0.0
    nan_mask = (depth_img_rescaled != depth_img_rescaled)
    depth_img_rescaled[nan_mask] = kZeroValue # set nan pixels to 0

    return depth_img_rescaled

class SegmentDepthWrapper(threading.Thread):

    def __init__(self, depth_segmentor, depth_img,  rgb_img):
        threading.Thread.__init__(self)

        self.depth_img_scaled = preprocess(depth_img)
        self.depth_segmentor = depth_segmentor
        self.rgb_img = rgb_img.astype(np.float32)
    
        self.depth_map = None
        self.segment_masks_list = []

    def run(self):
        self.depth_segmentor.depthSegment(self.depth_img_scaled,
                                          self.rgb_img)
        self.depth_map = self.depth_segmentor.get_depthMap()
        self.segment_masks_list = self.depth_segmentor.get_segmentMasks()

class Segment:
    def __init__(self, points, is_thing, instance_label, class_label, \
            inst_confidence, overlap_ratio, pose_confidence,  \
            pose, index, center = None, segment_label = 0):

        self.points = points.astype(np.float32).reshape(-1,3)
        self.is_thing = is_thing
        self.instance_label = np.float32(instance_label)
        self.class_label= class_label
        self.inst_confidence = np.float32(inst_confidence)
        self.overlap_ratio = np.float32(overlap_ratio)
        self.pose = pose.astype(np.float32)
        self.index = index
        self.pose_confidence = pose_confidence
        self.segment_label = segment_label

        if (center is None) or (np.array(center).shape != (1,3)):  
            self.center = np.mean(self.points, axis=0).astype(np.float32).reshape(1,3)
        else:
            self.center = center.astype(np.float32).reshape(1,3)
        self.box_points = np.zeros((1,3))

    # def calculateConfidenceDefault(self, weight=0.5):
    #     self.geometry_confidence = np.ones((1,self.points.shape[0])).reshape(1,-1).astype(np.float32)

    def calculateBBox(self, voxel_grid = 0.02, sampling_dist = 0.025):
        seg_pcl = pcl.PointCloud(self.points)
        # sparsify pcl
        pcl_sparse_filter = seg_pcl.make_voxel_grid_filter()
        pcl_sparse_filter.set_leaf_size(voxel_grid,voxel_grid,voxel_grid)
        seg_pcl_voxel = pcl_sparse_filter.filter()
        # remove outlier
        pcl_filter = seg_pcl_voxel.make_statistical_outlier_filter()
        pcl_filter.set_mean_k (10)
        pcl_filter.set_std_dev_mul_thresh (2.0)
        pcl_filtered =  pcl_filter.filter()
        # calculate boundiing box
        Bbox_extractor = pcl.MomentOfInertiaEstimation()
        Bbox_extractor.set_InputCloud(pcl_filtered)
        Bbox_extractor.compute()
        min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB = Bbox_extractor.get_OBB()

        len_wid_hei_half = np.array(max_point_OBB).reshape(-1)
        sampling_num_x = int( max( np.floor(len_wid_hei_half[0]*2/sampling_dist -1), 1) )
        sampling_num_y = int( max( np.floor(len_wid_hei_half[1]*2/sampling_dist -1), 1) )
        sampling_num_z = int( max( np.floor(len_wid_hei_half[2]*2/sampling_dist -1), 1) )

        Bbox_vertices_unit = [
            [0,0,0], # box center
            [1,1,1], # v1
            [-1,1,1], # v2
            [-1,1,-1], # v3
            [1,1,-1], # v4
            [1,-1,1], # v5
            [-1,-1,1], # v6
            [-1,-1,-1], # v7
            [1,-1,-1] # v8
        ]
        x_sample_range = np.interp(np.array(range(sampling_num_x)), [0,sampling_num_x], [-1,1])
        y_sample_range = np.interp(np.array(range(sampling_num_y)), [0,sampling_num_y], [-1,1])
        z_sample_range = np.interp(np.array(range(sampling_num_z)), [0,sampling_num_z], [-1,1])

        Bbox_vertices_unit.extend([[x_sample, 1, 1] for x_sample in x_sample_range]) # samples between v1-v2
        Bbox_vertices_unit.extend([[-1, 1, z_sample] for z_sample in z_sample_range]) # samples between v2-v3
        Bbox_vertices_unit.extend([[x_sample, 1, -1] for x_sample in x_sample_range]) # samples between v3-v4
        Bbox_vertices_unit.extend([[1, 1, z_sample] for z_sample in z_sample_range]) # samples between v4-v1
        Bbox_vertices_unit.extend([[1, y_sample, 1] for y_sample in y_sample_range]) # samples between v1-v5
        Bbox_vertices_unit.extend([[-1, y_sample, 1] for y_sample in y_sample_range]) # samples between v2-v6
        Bbox_vertices_unit.extend([[-1, y_sample, -1] for y_sample in y_sample_range]) # samples between v3-v7
        Bbox_vertices_unit.extend([[1, y_sample, -1] for y_sample in y_sample_range]) # samples between v4-v8
        Bbox_vertices_unit.extend([[x_sample, -1, 1] for x_sample in x_sample_range]) # samples between v5-v6
        Bbox_vertices_unit.extend([[-1, -1, z_sample] for z_sample in z_sample_range]) # samples between v6-v7
        Bbox_vertices_unit.extend([[x_sample, -1, -1] for x_sample in x_sample_range]) # samples between v7-v8
        Bbox_vertices_unit.extend([[1, -1, z_sample] for z_sample in z_sample_range]) # samples between v8-v5
        Bbox_vertices_unit = np.array(Bbox_vertices_unit).reshape(-1,3)

        Bbox_vertices_AA = Bbox_vertices_unit*np.repeat(np.abs(max_point_OBB), Bbox_vertices_unit.shape[0], axis=0) # (n, 3)
        Bbox_vertices_Orient =  (Bbox_vertices_AA@rotational_matrix_OBB.T) + position_OBB.reshape(1,3)
        self.box_points = (Bbox_vertices_Orient).reshape(-1,3).astype(np.float32) # (n, 3)

class SegmentsGenerator:
    def __init__(self, gsm_node, depth_segmentor, panoptic_segmentor, \
        save_resutls_img=False, img_folder = None, \
        save_segments = False, use_segments = False, segments_folder = None,
        save_panoptics = False, use_panoptics = False, panoptics_folder = None,
        save_geometrics = False, use_geometrics = False, geometrics_folder = None):
        
        self.depth_segmentor = depth_segmentor
        self.panoptic_segmentor = panoptic_segmentor

        self.save_resutls_img = save_resutls_img
        self.img_folder = img_folder
        if self.save_resutls_img:
            self.semantic_folder = os.path.join(self.img_folder, 'panoptic_seg')
            if not os.path.exists(self.semantic_folder):
                os.makedirs(self.semantic_folder)

        self.save_segments = save_segments
        self.use_segments =use_segments
        self.segments_folder = segments_folder
        if(self.save_segments and self.segments_folder is not None):
            if not os.path.exists(self.segments_folder):
                os.makedirs(self.segments_folder)
        else:
            self.save_segments = False

        self.save_panoptics = save_panoptics
        self.use_panoptics =use_panoptics
        self.panoptics_folder = panoptics_folder
        if(self.save_panoptics and self.panoptics_folder is not None):
            if not os.path.exists(self.panoptics_folder):
                os.makedirs(self.panoptics_folder)
        else:
            self.save_panoptics = False

        self.save_geometrics = save_geometrics
        self.use_geometrics =use_geometrics
        self.geometrics_folder = geometrics_folder
        if(self.save_geometrics and self.geometrics_folder is not None):
            if not os.path.exists(self.geometrics_folder):
                os.makedirs(self.geometrics_folder)
        else:
            self.save_geometrics = False

        return None

    def Segmennt2D(self, depth_img, rgb_img, frame_i):
        result = None
        depth_segmentor_thread = SegmentDepthWrapper(self.depth_segmentor,
                                                      depth_img,
                                                      rgb_img)
        # depth segmentation subprocess
        depth_segmentor_thread.start()

        # panoptic segmentation
        panoptic_result = None
        if self.panoptic_segmentor.use_gt:
            panoptic_result = self.panoptic_segmentor.forward(frame_i)
        else:
            panoptic_result = self.panoptic_segmentor.forward(rgb_img)
        if len(panoptic_result['info']) == 0:
            depth_segmentor_thread.join()
            return result

        # waiting for depth seg result
        depth_segmentor_thread.join()
        depth_map = depth_segmentor_thread.depth_map
        segment_masks =  depth_segmentor_thread.segment_masks_list
        if len(segment_masks) == 0:
            return result
        # extract instance/stuff information 
        id2info_instance = {}
        id2info_stuff = {}
        seg_map = panoptic_result['seg_map']

        for id_info in panoptic_result['info']:
            id = id_info['id']
            is_thing = id_info['isthing']
            if is_thing:
                # instance
                id2info_instance[id] = id_info
            else:
                # stuff
                id2info_stuff[id] = id_info

        result = {'seg_map': seg_map, 'id2info_instance':id2info_instance, 'id2info_stuff':id2info_stuff, \
                    'segment_masks': segment_masks, 'depth_map': depth_map}

        if( self.save_panoptics and (self.panoptics_folder is not None) and
            self.save_geometrics and (self.geometrics_folder is not None)):
            self.save2DPanopticGeometricSegs(seg_map, panoptic_result['info'], segment_masks, frame_i)
        return result
    
    def save2DPanopticGeometricSegs(self, seg_map, panoptic_info, 
                                    segment_masks, frame_i):
        # save panoptic information
        panoptic_mask = seg_map
        panoptic_mask_f = os.path.join(self.panoptics_folder, str(frame_i).zfill(5)+"_mask.png")
        cv2.imwrite(panoptic_mask_f, panoptic_mask)

        panoptic_info_dict = {
            'ids': [sem_info['id'] for sem_info in panoptic_info],
            "is_thing": [sem_info['isthing'] for sem_info in panoptic_info],
            "cates": [sem_info['category_id'] for sem_info in panoptic_info],
            "areas": [sem_info['area'] for sem_info in panoptic_info]
        }
        panoptic_info_f = os.path.join(self.panoptics_folder, str(frame_i).zfill(5)+"_info.h5")
        dictToHd5(panoptic_info_f, panoptic_info_dict)

        # save geometric information
        geometric_seg_mask = np.zeros(panoptic_mask.shape, dtype=np.uint8)
        for seg_idx in range(len(segment_masks)):
            geometric_seg_mask[segment_masks[seg_idx].astype(bool)] = seg_idx+1
        geometric_seg_mask_f = os.path.join(self.geometrics_folder, str(frame_i).zfill(5)+"_mask.png")
        cv2.imwrite(geometric_seg_mask_f, geometric_seg_mask)

    def load2DPanopticGeometricSegs(self, camera_K, depth_img_scaled, frame_i):
        # load panoptic information
        panoptic_mask_f = os.path.join(self.panoptics_folder, str(frame_i).zfill(5)+"_mask.png")
        panoptic_info_f = os.path.join(self.panoptics_folder, str(frame_i).zfill(5)+"_info.h5")
        if( (not os.path.isfile(panoptic_mask_f)) or 
            (not os.path.isfile(panoptic_info_f))):
            return None
        panoptic_info_dict = hd5ToDict(panoptic_info_f)
        id2info = {}
        id2info_instance = {}
        id2info_stuff = {}
        for sem_idx in range(len(panoptic_info_dict['ids'])):
            id = panoptic_info_dict['ids'][sem_idx]
            is_thing = panoptic_info_dict['is_thing'][sem_idx]
            sem_info = {
                'id': id, 'isthing':is_thing, 
                'category_id': panoptic_info_dict['cates'][sem_idx],
                'area': panoptic_info_dict['areas'][sem_idx]
            }
            if is_thing:
                # instance
                id2info_instance[id] = sem_info
            else:
                # stuff
                id2info_stuff[id] = sem_info

            id2info[id] = sem_info

        panoptic_mask = cv2.imread(panoptic_mask_f, cv2.IMREAD_UNCHANGED)

        # load geometric information
        geometric_seg_mask_f = os.path.join(self.geometrics_folder, str(frame_i).zfill(5)+"_mask.png")
        if( (not os.path.isfile(geometric_seg_mask_f)) ):
            return None
        geometric_seg_mask = cv2.imread(geometric_seg_mask_f, cv2.IMREAD_UNCHANGED)

        segment_masks = []
        segs_ids = np.unique(geometric_seg_mask)
        for seg_id in segs_ids:
            if seg_id == 0:
                continue
            seg_mask = np.zeros(panoptic_mask.shape, dtype=bool)
            seg_mask[geometric_seg_mask==seg_id] = True
            segment_masks.append(seg_mask)
        
        # get depth map
        depth_map = cv2.rgbd.depthTo3d(depth=depth_img_scaled,K=camera_K)
        result = {'seg_map': panoptic_mask, 
                  'id2info': id2info,
                  'id2info_instance': id2info_instance, 
                  'id2info_stuff': id2info_stuff,
                  'segment_masks': np.array(segment_masks).reshape(-1,panoptic_mask.shape[0], panoptic_mask.shape[-1]), 
                  'depth_map': depth_map}
        
        return result
    
    def generateSegments(self, seg_result_2D, pose, pose_confidence, frame_i):
        # get panoptic segmentation result
        segment_list = []
        segment_masks = seg_result_2D['segment_masks']
        seg_map = seg_result_2D['seg_map']
        id2info_instance = seg_result_2D['id2info_instance']
        id2info_stuff = seg_result_2D['id2info_stuff']
        ## get panoptic seg masks
        num_panoptic_segs = len(id2info_instance)+len(id2info_stuff)
        panop_seg_masks = np.zeros((seg_map.shape[0], seg_map.shape[1], num_panoptic_segs+1), dtype=bool)
        mask_idxs_2D = np.indices(seg_map.shape)
        panop_seg_masks[mask_idxs_2D[0], mask_idxs_2D[1], seg_map] = True

        # generate segments candidates
        sem_depth_segments = []
        extra_instances = []
        background_segments = []
        for mask_i in range(segment_masks.shape[0]):
            depth_seg_mask = segment_masks[mask_i,:,:].copy()
            depth_seg_mask = depth_seg_mask.astype(bool)
            # remove small segments
            if np.sum(depth_seg_mask) < 100:
                continue
            depth_sem_seg_ids = seg_map[depth_seg_mask].reshape(-1)
            depth_seg_area = depth_sem_seg_ids.shape[0]
            candidate_pairs = Counter(depth_sem_seg_ids)

            max_overlap_area = 0
            max_candidate_id = 0 
            for panoptic_id in candidate_pairs:
                if(panoptic_id == 0):
                    continue
                candidate_area = candidate_pairs[panoptic_id]
                is_thing = (panoptic_id in id2info_instance)
                is_stuff = (panoptic_id in id2info_stuff)
                if is_thing:
                    # if(candidate_area>0.9*id2info_instance[panoptic_id]['area'] and candidate_area<0.5*depth_seg_area): # Han et al
                    if(candidate_area>0.01*id2info_instance[panoptic_id]['area'] and candidate_area<0.8*depth_seg_area): # Tune9 ours
                    # if(False): # voxbloxpp
                    # if depth-undersegment, then further seg it 
                        extracted_mask = np.logical_and(depth_seg_mask, panop_seg_masks[:,:,panoptic_id])
                        overlap_ratio = candidate_area * 1.0 / id2info_instance[panoptic_id]['area']
                        inst_score = 1.0 if is_thing else 0.5
                        extra_instances.append({'mask': extracted_mask, 'id': panoptic_id, 'is_thing': True, \
                            'inst_score': inst_score, 'overlap_r':overlap_ratio })
                        # further determine remaining part
                        depth_seg_area -= candidate_area
                        depth_seg_mask[extracted_mask] = False
                    else:
                        if max_overlap_area<candidate_area:
                            max_overlap_area = candidate_area
                            max_candidate_id = panoptic_id
                    continue
                elif is_stuff:
                    # if(False): # voxbloxpp, Han et al
                    if(candidate_area>0.05*id2info_stuff[panoptic_id]['area'] and candidate_area<0.8*depth_seg_area): # Tune9 ours
                    # if depth-undersegment, then further seg it 
                        extracted_mask = np.logical_and(depth_seg_mask, panop_seg_masks[:,:,panoptic_id])
                        overlap_ratio = candidate_area * 1.0 / id2info_stuff[panoptic_id]['area']
                        inst_score = 1.0 if is_thing else 0.5
                        extra_instances.append({'mask': extracted_mask, 'id': panoptic_id, 'is_thing': False, \
                            'inst_score': inst_score, 'overlap_r':overlap_ratio })
                        # further determine remaining part
                        depth_seg_area -= candidate_area
                        depth_seg_mask[extracted_mask] = False
                    else:
                        if max_overlap_area<candidate_area:
                            max_overlap_area = candidate_area
                            max_candidate_id = panoptic_id
                    continue
            # determine semantic label for depth_seg
            if(max_overlap_area>=0.2*depth_seg_area):
                overlap_ratio = max_overlap_area*1.0/depth_seg_area
                is_thing = (max_candidate_id in id2info_instance)
                # inst_score = id2info_instance[max_candidate_id]['score'] if is_thing else 0.5
                inst_score = 1.0 if is_thing else 0.5
                sem_depth_segments.append({'mask': depth_seg_mask, 'id': max_candidate_id, 'is_thing': is_thing, \
                    'inst_score': inst_score, 'overlap_r': overlap_ratio})
            else:
                overlap_ratio = candidate_pairs[0] * 1.0 / depth_seg_area
                background_segments.append({'mask': depth_seg_mask, 'id': 0, 'is_thing': False, 'inst_score': 0.5 \
                    , 'overlap_r': overlap_ratio})
        # generate segments
        mask_segments_singleframe = np.zeros_like(seg_map, dtype=np.uint8)
        depth_map = seg_result_2D['depth_map']
        seg_index = 0
        for info_sem_depth_seg in sem_depth_segments:
            points = depth_map[info_sem_depth_seg['mask']].astype(np.float32).reshape(-1,3)
            is_thing = info_sem_depth_seg['is_thing']
            instance_label = info_sem_depth_seg['id']
            semantic_label = id2info_instance[instance_label]['category_id'] if is_thing else id2info_stuff[instance_label]['category_id']
            semantic_label = semantic_map(semantic_label)
            inst_score = info_sem_depth_seg['inst_score']
            overlap_ratio = info_sem_depth_seg['overlap_r']
            segment = Segment(points, is_thing, instance_label, semantic_label, inst_score, overlap_ratio, pose_confidence, pose, seg_index)
            # segment.calculateConfidenceDefault()
            segment_list.append(segment)
            seg_index += 1
            mask_segments_singleframe[info_sem_depth_seg['mask']] = seg_index
        for extected_instance_seg in extra_instances:
            # x1, y1, x2, y2 = extected_instance_seg['box']
            points = depth_map[extected_instance_seg['mask']].astype(np.float32).reshape(-1,3)
            is_thing = extected_instance_seg['is_thing']
            instance_label = extected_instance_seg['id']
            semantic_label = BackgroundSemId
            if is_thing:
                semantic_label = id2info_instance[instance_label]['category_id']
            else:
                semantic_label = id2info_stuff[instance_label]['category_id']
            semantic_label = semantic_map(semantic_label)
            inst_score = extected_instance_seg['inst_score']
            overlap_ratio = extected_instance_seg['overlap_r']
            segment = Segment(points, is_thing, instance_label, semantic_label, inst_score, overlap_ratio, pose_confidence, pose, seg_index)
            # segment.calculateConfidenceDefault()
            segment_list.append(segment)
            seg_index += 1 
            mask_segments_singleframe[extected_instance_seg['mask']] = seg_index
        for background_seg in background_segments:
            points = depth_map[background_seg['mask']].astype(np.float32).reshape(-1,3)
            is_thing = background_seg['is_thing']
            instance_label = background_seg['id']
            semantic_label = BackgroundSemId # background semantic label
            inst_score = background_seg['inst_score']
            overlap_ratio = background_seg['overlap_r']
            segment = Segment(points, is_thing, instance_label, semantic_label, inst_score, overlap_ratio, pose_confidence, pose, seg_index)
            # segment.calculateConfidenceDefault()
            segment_list.append(segment)
            seg_index += 1
            mask_segments_singleframe[background_seg['mask']] = seg_index
        if self.save_segments:
            mask_f = os.path.join(self.segments_folder, str(frame_i).zfill(5)+"_mask.png")
            cv2.imwrite(mask_f, mask_segments_singleframe)
        return segment_list

    def outlierRemove(self, segment_list, neighbor_dist_th = 0.05):
        # TODO
        # instance_to_seg_pair = {}   
        # # get instance-segment map
        # for seg in segment_list:
        #     if seg.is_thing:
        #         if seg.instance_label in instance_to_seg_pair:
        #             instance_to_seg_pair[seg.instance_label].append(seg.index)
        #         else:
        #             instance_to_seg_pair[seg.instance_label] = [seg.index]

        # for instance_label in instance_to_seg_pair:
        #     # get neighbor map
        #     instance_seg_list = instance_to_seg_pair[instance_label]
        #     neighber_map = { seg_index:[] for seg_index in instance_seg_list}
        #     for i, seg_i in enumerate(instance_seg_list):
        #         for j in range(seg_i+1, )
        return segment_list


    def frameToSegments(self, depth_img, rgb_img, pose, pose_confidence, frame_i, camera_K = None):
        # TODEBUG
        t0 = time.time()
        segment_list = []

        seg_result_2D = None
        if self.use_panoptics and self.use_geometrics and (camera_K is not None):
            depth_img_scaled = cv2.rgbd.rescaleDepth(depth_img, cv2.CV_32FC1)
            seg_result_2D = self.load2DPanopticGeometricSegs(
                camera_K, depth_img_scaled, frame_i
            )
        else:
            seg_result_2D = self.Segmennt2D(depth_img, rgb_img, frame_i)
        if seg_result_2D is None:
            return segment_list # return if nothing from 2D segmentation
        
        segment_list = self.generateSegments(seg_result_2D, pose, pose_confidence, frame_i)

        # save segments information
        if self.save_segments:
            seg_info = {'is_thing':[], 'instance_label':[],'class_label':[], 'inst_confidence':[], \
                'overlap_ratio': [], 'pose':[], 'center':[], 'seg_num':0}
            for seg in segment_list:
                seg_info['is_thing'].append(seg.is_thing)
                seg_info['instance_label'].append(seg.instance_label)
                seg_info['class_label'].append(seg.class_label)
                seg_info['inst_confidence'].append(seg.inst_confidence)
                seg_info['overlap_ratio'].append(seg.overlap_ratio)
                seg_info['pose'].append(seg.pose)
                seg_info['pose_confidence'].append(seg.pose_confidence)
                seg_info['center'].append(seg.center)
            seg_info['seg_num'] = len(segment_list)
            seg_info_f = os.path.join(self.segments_folder, str(frame_i).zfill(5)+"_seg_info.h5")
            dictToHd5(seg_info_f, seg_info)

        # self.gsm_node.outputLog("   Seg Generation in python cost %f s" %(time.time() - t0))
        return segment_list

    def loadSegments(self, depth_scaled, camera_K, frame_i):

        segments_list = []
        mask_f = os.path.join(self.segments_folder, str(frame_i).zfill(5)+"_mask.png")
        mask = cv2.imread(mask_f, cv2.IMREAD_UNCHANGED)
        seg_info_f = os.path.join(self.segments_folder, str(frame_i).zfill(5)+"_seg_info.h5")
        if( (not os.path.isfile(mask_f)) or (not os.path.isfile(seg_info_f)) ):
            return segments_list

        seg_info= hd5ToDict(seg_info_f)
        seg_indexes = np.unique(mask)
        for seg_i in range(seg_info['seg_num']):
            seg_mask = (mask==(seg_i+1))
            points = cv2.rgbd.depthTo3d(depth=depth_scaled,K=camera_K,mask=seg_mask.astype(np.uint8))
            is_thing = seg_info['is_thing'][seg_i]
            instance_label = seg_info['instance_label'][seg_i]
            class_label = seg_info['class_label'][seg_i]
            inst_confidence = seg_info['inst_confidence'][seg_i]
            overlap_ratio = seg_info['overlap_ratio'][seg_i]
            pose_confidence = seg_info['pose_confidence'][seg_i]
            pose = seg_info['pose'][seg_i]
            center = seg_info['center'][seg_i]

            segment_label = 0
            if 'segment_label' in seg_info:
                segment_label = seg_info['segment_label'][seg_i]
            segment = Segment(points, is_thing, instance_label, class_label, 
                inst_confidence, overlap_ratio, pose_confidence, pose, seg_i, 
                center, segment_label)
            if(segment.points.shape[0] < 1):
                continue
            # segment.calculateConfidenceDefault()
            segments_list.append(segment)
        return segments_list

def dictToHd5(file, dict):
	f = h5py.File(file,'w')
	for key in dict:
		f[key] = dict[key]
	f.close() 
	return None

def hd5ToDict(file):
	f = h5py.File(file,'r')
	dict = {}
	for key in f:
		dict[key] = np.array(f[key])
	f.close() 
	return dict

def checkSegmentFramesEqual(segs_framesA, segs_framesB):
    if(len(segs_framesA)!=len(segs_framesB)):
        print(" Not Equal, length of segment lists")
        return False
    for f_i, seg_A in enumerate(segs_framesA):
        seg_B = segs_framesB[f_i]
        # print("     check seg num %d "%(f_i))
        if(not np.isclose(seg_A.pose,seg_B.pose).all()):
            print("    Not Equal pose in %d frame "%(f_i))
            return False
        if(not np.isclose(seg_A.center,seg_B.center).all()):
            print("    Not Equal center in %d frame "%(f_i))
            return False
        if(not np.isclose(seg_A.points,seg_B.points, atol=1e-4).all()):
            print("    Not Equal points in %d frame "%(f_i))
            return False
        if(not np.isclose(seg_A.inst_confidence,seg_B.inst_confidence).all()):
            print("    Not Equal label_confidence in %d frame "%(f_i))
            return False
        if(not np.isclose(seg_A.overlap_ratio,seg_B.overlap_ratio).all()):
            print("    Not Equal label_confidence in %d frame "%(f_i))
            return False
        if((seg_A.instance_label!=seg_B.instance_label)):
            print("    Not Equal instance_label in %d frame "%(f_i))
            return False
        if((seg_A.class_label!=seg_B.class_label)):
            print("    Not Equal class_label in %d frame "%(f_i))
            return False
        if((seg_A.is_thing!=seg_B.is_thing)):
            print("    Not Equal class_label in %d frame "%(f_i))
            return False
    return True

class DataLoader:
    def __init__(self, dir, traj_filename, preload_img = False, preload_depth = False):
        # whether to preload data into memory
        self.preload_img = preload_img
        self.preload_depth = preload_depth

        # parse data location
        self.dir = dir
        self.depth_folder = os.path.join(self.dir, "depth")
        self.depth_files = os.listdir(self.depth_folder)
        self.depth_files.sort()
        self.depth_indexes = [int(depth_f.split('.')[0]) for depth_f in self.depth_files]
        self.depth_path_map = {index: os.path.join(self.depth_folder, str(index)+".png") \
                                for index in self.depth_indexes}

        self.rgb_folder = os.path.join(self.dir, "color")
        self.rgb_files = os.listdir(self.rgb_folder)
        self.rgb_files.sort()
        self.rgb_indexes = [int(color_f.split('.')[0]) for color_f in self.rgb_files]
        self.rgb_path_map = {index: os.path.join(self.rgb_folder, str(index)+".jpg") \
                                for index in self.rgb_indexes}

        # load poses first
        self.traj_f = os.path.join(self.dir, traj_filename)
        self.readTrajectory()
        self.traj_indexes = list(self.poses.keys())

        # get frame indexs
        self.indexes = set.intersection( set(self.depth_indexes), set(self.rgb_indexes),  set(self.traj_indexes))
        self.indexes = list(self.indexes)
        self.indexes.sort()
        self.index_min = min(self.indexes)
        self.index_max = max(self.indexes)  

        # load pose confidence
        self.inlier_ratio_f = os.path.join(self.dir, "orbslam", "inlier_ratio.txt")
        self.uncertainty_f = os.path.join(self.dir, "orbslam", "uncertainty.txt")
        self.inlier_num_f = os.path.join(self.dir, "orbslam", "inlier_num.txt")
        self.calculatePoseConfidence()

        # get camera matrixs
        self.rgb_extrinsic_f = os.path.join(self.dir, "intrinsic", "extrinsic_color.txt")
        self.rgb_intrinsic_f = os.path.join(self.dir, "intrinsic", "intrinsic_color.txt")
        self.depth_extrinsic_f = os.path.join(self.dir, "intrinsic", "extrinsic_depth.txt")
        self.depth_intrinsic_f = os.path.join(self.dir, "intrinsic", "intrinsic_depth.txt")
        self.rgb_extrinsic = np.loadtxt(self.rgb_extrinsic_f)
        self.rgb_intrinsic = np.loadtxt(self.rgb_intrinsic_f)[:3, :3]
        self.depth_extrinsic = np.loadtxt(self.depth_extrinsic_f)
        self.depth_intrinsic = np.loadtxt(self.depth_intrinsic_f)[:3, :3]
        assert(np.isclose(np.eye(4), self.rgb_extrinsic).all())
        assert(np.isclose(np.eye(4), self.depth_extrinsic).all())
        self.homograph_color_to_depth = self.depth_intrinsic @ np.linalg.inv(self.rgb_intrinsic)

        # get depth image shape 
        depth_f = self.depth_path_map[self.indexes[0]]
        depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
        self.depth_h = depth_img.shape[0]
        self.depth_w = depth_img.shape[1]

        # preload data in RAM
        if self.preload_img:
            self.images = {}
            for idx in self.indexes:
                image_f = self.rgb_path_map[idx]
                rgb_img = cv2.imread(image_f,cv2.IMREAD_UNCHANGED)
                rgb_img_aligned = cv2.warpPerspective(rgb_img, self.homograph_color_to_depth,
                    (self.depth_w, self.depth_h) )
                self.images[idx] = rgb_img_aligned

        if self.preload_depth:
            self.depths = {}
            for idx in self.indexes:
                depth_f = self.depth_path_map[idx]
                depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
                self.depths[idx] = depth_img

    def readTrajectory(self):
        self.poses = {}
        f = open(self.traj_f,'r')
        T_WC = []
        current_id = None
        for line in f.readlines():
            data = line.split(' ')
            if(len(data) == 3):
                if T_WC:
                    T_WC = np.array(T_WC)
                    r = Rotation.from_matrix(T_WC[:3,:3])
                    T_WC[:3,:3] = r.as_matrix()
                    self.poses[current_id] = np.array(T_WC).reshape(4,4)
                current_id = int(data[0])
                T_WC = []

            elif(len(data) == 4):
                T_WC.append([float(data[0]),float(data[1]),float(data[2]),float(data[3])])
        f.close()
        
    def calculatePoseConfidence(self):
        # inlier_ratio_arr = np.loadtxt(self.inlier_ratio_f).reshape(-1,2)
        # self.inlier_ratio_map = {}
        # for arr_idx in range(inlier_ratio_arr.shape[0]):
        #     self.inlier_ratio_map[int(inlier_ratio_arr[arr_idx,0])] = inlier_ratio_arr[arr_idx,1]
        
        # # calculate pose confidence, try inlier ratio first 
        # self.pose_confidence_map = {}
        # for index in self.indexes:
        #     if index == self.index_min:
        #         self.pose_confidence_map[index] = 1.0
        #         continue
        #     if index not in self.inlier_ratio_map:
        #         self.pose_confidence_map[index] = 0.5
        #         continue
        #     else:
        #         self.pose_confidence_map[index] = self.inlier_ratio_map[index]
            
        # uncertainty_arr = np.loadtxt(self.uncertainty_f).reshape(-1,2)
        # # uncertainty_arr = uncertainty_arr[:,1]
        # mean = np.mean(uncertainty_arr[:,1])
        # std = np.std(uncertainty_arr[:,1])
        
        # self.uncertainty_map = {}
        # for arr_idx in range(uncertainty_arr.shape[0]):
        #     self.uncertainty_map[int(uncertainty_arr[arr_idx,0])] = uncertainty_arr[arr_idx,1]
        
        # # calculate pose confidence, try inlier ratio first 
        # self.pose_confidence_map = {}
        # for index in self.indexes:
        #     if index == self.index_min:
        #         self.pose_confidence_map[index] = 1.0
        #         continue
        #     if index not in self.uncertainty_map:
        #         self.pose_confidence_map[index] = 1.0
        #         continue
        #     else:
        #         confidence = 0.8 + (self.uncertainty_map[index] - mean)/(2*std)
        #         if confidence < 0.0:
        #             confidence = 0.0
        #         if confidence > 1.0:
        #             confidence = 1.0
        #         self.pose_confidence_map[index] = confidence
        
        if not os.path.isfile(self.inlier_num_f):
            self.pose_confidence_map = {}
            for index in self.indexes:
                self.pose_confidence_map[index] = 1.0
            return
        inlier_num_arr = np.loadtxt(self.inlier_num_f) 
        self.inlier_num = {}
        for arr_idx in range(inlier_num_arr.shape[0]):
            self.inlier_num[int(inlier_num_arr[arr_idx,0])] = inlier_num_arr[arr_idx,1]
        # calculate pose confidence, try scaled inlier num 
        self.pose_confidence_map = {}
        for index in self.indexes:
            if index == self.index_min:
                self.pose_confidence_map[index] = 2.0
                continue
            if index not in self.inlier_num:
                self.pose_confidence_map[index] = 1.0
                continue
            else:
                confidence = 1.0 + self.inlier_num[index] /1000.0
                confidence = min(confidence, 3.0)
                self.pose_confidence_map[index] = confidence
                
        breakpoint =  None
    def readPoseConfidence(self, index):
        if index not in self.pose_confidence_map:
            return None
        else:   
            return self.pose_confidence_map[index]
        
    def getPoseFromIndex(self, index):
        pose = self.poses[index]
        return pose

    def getDataFromIndex(self, index):
        if index == -1:
            index = self.index_min
        # normally start from 0: 0.; 0-indexed
        if(index not in self.indexes):
            return None,None,None

        rgb_img_aligned = None
        depth_img = None
        pose = None

        if self.preload_img:
            rgb_img_aligned = self.images[index]
        else:
            image_f = self.rgb_path_map[index]
            rgb_img = cv2.imread(image_f,cv2.IMREAD_UNCHANGED)
            rgb_img_aligned = cv2.warpPerspective(rgb_img, self.homograph_color_to_depth,
                (self.depth_w, self.depth_h) )

        if self.preload_depth:
            depth_img = self.depths[index]
        else:
            depth_f = self.depth_path_map[index]
            depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
        
        pose = self.poses[index]
        # check validity of pose
        is_pose_valid = self.isPoseValid(pose)
        if not is_pose_valid:
            return None,None,None

        return rgb_img_aligned, depth_img, pose.astype(np.float32)

    def isPoseValid(self, pose):
        is_nan = np.isnan(pose).any() or np.isinf(pose).any()
        if is_nan:
            return False

        R_matrix = pose[:3, :3]
        I = np.identity(3)
        is_rotation_valid = ( np.isclose( np.matmul(R_matrix, R_matrix.T), I , atol=1e-3) ).all and np.isclose(np.linalg.det(R_matrix) , 1, atol=1e-3)
        if not is_rotation_valid:
            return False

        return True

    def getDepthScaledFromIndex(self, index):
        if index == -1:
            index = self.index_min
        if(index not in self.indexes):
            return None

        depth_img = None
        if self.preload_depth:
                depth_img = self.depths[index]
        else:
            depth_f = self.depth_path_map[index]
            depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)

        depth_img_scaled = cv2.rgbd.rescaleDepth(depth_img, cv2.CV_32FC1)
        return depth_img_scaled

    def getCameraMatrix(self):
        # return depth camera matrix 
        return self.depth_intrinsic.astype(np.float32)

class DataLoaderSceneNN:
    def __init__(self, dir, traj_filename, preload_img = False, preload_depth = False):
        # whether to preload data into memory
        self.preload_img = preload_img
        self.preload_depth = preload_depth

        # parse data location
        self.dir = dir
        self.depth_folder = os.path.join(self.dir, "depth")
        self.depth_files = os.listdir(self.depth_folder)
        self.depth_files.sort()
        self.depth_indexes = [int(depth_f[5:10])-1 for depth_f in self.depth_files]
        self.depth_path_map = {index: os.path.join(self.depth_folder, "depth"+str(index+1).zfill(5)+".png") \
                                for index in self.depth_indexes}

        self.rgb_folder = os.path.join(self.dir, "image")
        self.rgb_files = os.listdir(self.rgb_folder)
        self.rgb_files.sort()
        self.rgb_indexes = [int(color_f[5:10])-1 for color_f in self.rgb_files]
        self.rgb_path_map = {index: os.path.join(self.rgb_folder, "image"+str(index+1).zfill(5)+".png") \
                                for index in self.rgb_indexes}

        # load poses first
        self.traj_f = os.path.join(self.dir, traj_filename)
        self.readTrajectory()
        self.traj_indexes = list(self.poses.keys())

        # get frame indexs
        self.indexes = set.intersection( set(self.depth_indexes), set(self.rgb_indexes),  set(self.traj_indexes))
        self.indexes = list(self.indexes)
        self.indexes.sort()
        self.index_min = min(self.indexes)
        self.index_max = max(self.indexes)  

        # # get camera matrixs
        # self.rgb_extrinsic_f = os.path.join(self.dir, "intrinsic", "extrinsic_color.txt")
        # self.rgb_intrinsic_f = os.path.join(self.dir, "intrinsic", "intrinsic_color.txt")
        # self.depth_extrinsic_f = os.path.join(self.dir, "intrinsic", "extrinsic_depth.txt")
        # self.depth_intrinsic_f = os.path.join(self.dir, "intrinsic", "intrinsic_depth.txt")
        # self.rgb_extrinsic = np.loadtxt(self.rgb_extrinsic_f)
        # self.rgb_intrinsic = np.loadtxt(self.rgb_intrinsic_f)[:3, :3]
        # self.depth_extrinsic = np.loadtxt(self.depth_extrinsic_f)
        # self.depth_intrinsic = np.loadtxt(self.depth_intrinsic_f)[:3, :3]
        # assert(np.isclose(np.eye(4), self.rgb_extrinsic).all())
        # assert(np.isclose(np.eye(4), self.depth_extrinsic).all())
        # self.homograph_color_to_depth = self.depth_intrinsic @ np.linalg.inv(self.rgb_intrinsic)

        # get depth image shape 
        depth_f = self.depth_path_map[self.indexes[0]]
        depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
        self.depth_h = depth_img.shape[0]
        self.depth_w = depth_img.shape[1]

        # # preload data in RAM
        # if self.preload_img:
        #     self.images = {}
        #     for idx in self.indexes:
        #         image_f = self.rgb_path_map[idx]
        #         rgb_img = cv2.imread(image_f,cv2.IMREAD_UNCHANGED)
        #         rgb_img_aligned = cv2.warpPerspective(rgb_img, self.homograph_color_to_depth,
        #             (self.depth_w, self.depth_h) )
        #         self.images[idx] = rgb_img_aligned

        # if self.preload_depth:
        #     self.depths = {}
        #     for idx in self.indexes:
        #         depth_f = self.depth_path_map[idx]
        #         depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
        #         self.depths[idx] = depth_img

    def readTrajectory(self):
        self.poses = {}
        f = open(self.traj_f,'r')
        T_WC = []
        current_id = None
        for line in f.readlines():
            data = line.split(' ')
            if(len(data) == 3):
                if T_WC:
                    T_WC = np.array(T_WC)
                    r = Rotation.from_matrix(T_WC[:3,:3])
                    T_WC[:3,:3] = r.as_matrix()
                    self.poses[current_id] = np.array(T_WC).reshape(4,4)
                current_id = int(data[0])
                T_WC = []

            elif(len(data) == 4):
                T_WC.append([float(data[0]),float(data[1]),float(data[2]),float(data[3])])
        f.close()

    def getPoseFromIndex(self, index):
        pose = self.poses[index]
        return pose

    def getDataFromIndex(self, index):
        # normally start from 0: 0.; 0-indexed
        if(index not in self.indexes):
            return None,None,None

        depth_img = None
        pose = None

        if self.preload_img:
            rgb_img_aligned = self.images[index]
        else:
            image_f = self.rgb_path_map[index]
            rgb_img = cv2.imread(image_f,cv2.IMREAD_UNCHANGED)

        if self.preload_depth:
            depth_img = self.depths[index]
        else:
            depth_f = self.depth_path_map[index]
            depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
        
        pose = self.poses[index]
        # check validity of pose
        is_pose_valid = self.isPoseValid(pose)
        if not is_pose_valid:
            return None,None,None

        return rgb_img, depth_img, pose.astype(np.float32)

    def isPoseValid(self, pose):
        is_nan = np.isnan(pose).any() or np.isinf(pose).any()
        if is_nan:
            return False

        R_matrix = pose[:3, :3]
        I = np.identity(3)
        is_rotation_valid = ( np.isclose( np.matmul(R_matrix, R_matrix.T), I , atol=1e-3) ).all and np.isclose(np.linalg.det(R_matrix) , 1, atol=1e-3)
        if not is_rotation_valid:
            return False

        return True

    def getDepthScaledFromIndex(self, index):
        if(index not in self.indexes):
            return None

        depth_img = None
        if self.preload_depth:
                depth_img = self.depths[index]
        else:
            depth_f = self.depth_path_map[index]
            depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)

        depth_img_scaled = cv2.rgbd.rescaleDepth(depth_img, cv2.CV_32FC1)

        return depth_img_scaled

    def getCameraMatrix(self):
        # return depth camera matrix 
        K = np.array([[544.47329,0,320],[0,544.47329,240],[0,0,1]])
        return K.astype(np.float32)

def getHostId(gpu_id):
    HOST = str(gpu_id) + ".0.0."+str(gpu_id)
    return HOST
def getGpuDevice(gpu_id, gpu_num):
    assert(gpu_id >= 0)
    assert(gpu_num > gpu_id)
    if(gpu_num == 1):
        return "cuda"
    return "cuda:"+str(gpu_id)
def isPoseValid( pose):
    is_nan = np.isnan(pose).any() or np.isinf(pose).any()
    if is_nan:
        return False

    R_matrix = pose[:3, :3]
    I = np.identity(3)
    is_rotation_valid = ( np.isclose( np.matmul(R_matrix, R_matrix.T), I , atol=1e-3) ).all and np.isclose(np.linalg.det(R_matrix) , 1, atol=1e-3)
    if not is_rotation_valid:
        return False

    return True