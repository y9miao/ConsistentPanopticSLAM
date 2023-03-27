import numpy as np
import depth_seg_utils
import semantic_seg_utils

class Segment:
    def __init__(self, depth_map, rgb_image, pose, mask):

        self.mask = mask
        self.rgb_image = rgb_image.astype(np.uint8)
        self.depth_map = depth_map.astype(np.float32)
        self.pose = pose.astype(np.float32)

        self.instance_label=0
        self.class_label=0
        self.label_confidence=1
        self.geometry_confidence = None

    def labelSegments(self, semantic_results, overlap_th = 0.8):
        classes = semantic_results['class_ids']
        instances_masks = semantic_results['masks']
        scores = semantic_results['scores']

        max_overlap_times_score = 0.0
        # search instances for the one with max overlap
        for instance_i in range(len(classes)):
            instance_mask = instances_masks[:,:,instance_i]
            instance_score = scores[instance_i]

            overlap = np.sum(np.logical_and(self.mask, instance_mask))
            overlap_normalized = overlap*1.0/np.sum(self.mask!=0)
            overlap_times_score = overlap_normalized+instance_score

            if(overlap_times_score>max_overlap_times_score and overlap_normalized>overlap_th):
                max_overlap_times_score = overlap_times_score
                self.instance_label = instance_i + 1
                self.class_label = classes[instance_i]

        if(self.instance_label!=0):
            self.label_confidence = np.float32(max_overlap_times_score)

    def generatePoints(self, geometry_confidence):
        self.geometry_confidence = geometry_confidence[self.mask!=0].astype(np.float32)
        self.points = self.depth_map[self.mask!=0].astype(np.float32).reshape(-1,3)
        self.colors = self.rgb_image[self.mask!=0].astype(np.uint8).reshape(-1,3)
        # print("self.geometry_confidence.shape: ", self.geometry_confidence.shape)
        # print("self.points.shape: ", self.points.shape)
        # print("self.colors.shape: ", self.colors.shape)

        geometry_confidence_valid = (self.geometry_confidence!=0)
        # print("geometry_confidence_valid.shape: ", geometry_confidence_valid.shape)

        points_valid = np.logical_and(np.isfinite(self.points[:,0]),np.isfinite(self.points[:,1]))
        points_valid = np.logical_and(points_valid,np.isfinite(self.points[:,2]))
        points_valid = np.logical_and(points_valid,geometry_confidence_valid)

        # print("points_valid.shape: ", points_valid.shape)

        self.geometry_confidence = self.geometry_confidence[points_valid].reshape(-1,1)
        self.points = self.points[points_valid,:]
        self.colors = self.colors[points_valid,:]

        # print("self.geometry_confidence.shape: ", self.geometry_confidence.shape)
        # print("self.points.shape: ", self.points.shape)
        # print("self.colors.shape: ", self.colors.shape)

def frame2Segments(depth_segmentor, semantic_segmentor, depth_img, rgb_img, pose, save_resutls=False):
    segments_list = []

    # depth segmentation
    depth_img_scaled = depth_seg_utils.preprocess(depth_img)
    depth_segmentor.depthSegment(depth_img_scaled,rgb_img.astype(np.float32))
    depth_map = depth_segmentor.get_depthMap()
    normal_map = depth_segmentor.get_normalMap()
    # discontinuity_map = depth_segmentor.get_discontinuityMap()
    # distance_map = depth_segmentor.get_maxDistanceMap()
    # convexity_map = depth_segmentor.get_minConvexMap()
    # edge_map = depth_segmentor.get_edgeMap()
    
    segment_masks =  depth_segmentor.get_segmentMasks()
    cos_normal_angle, depth_confidence_map = \
        depth_seg_utils.confidence_calculation(depth_img_scaled,depth_map,normal_map)
    geometry_confidence_map = \
        depth_seg_utils.geometry_confidence_calculation(cos_normal_angle, depth_confidence_map)

    # semantic segmentation
    semantic_result = semantic_segmentor.forward(rgb_img)
    
    # generate segments
    for segment_i in range(segment_masks.shape[0]):
        seg = Segment(depth_map,rgb_img,pose,segment_masks[segment_i,:,:])
        seg.labelSegments(semantic_result)
        seg.generatePoints(geometry_confidence_map)
        segments_list.append(seg)

    if(save_resutls):
        label_map = depth_segmentor.get_labelMap()
        semantic_vis = semantic_segmentor._visualize(semantic_result, rgb_img)
        return segments_list, semantic_vis,label_map
    else:
        return segments_list, None, None
        


        
        