import sys
import os
import time
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import semantics.semantic_utils as semantic_utils

def parse_args():
    data_path = os.environ.get('DATA')
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # dataset 
    parse.add_argument("--dataset", type=str, default="scenenn", help="which data set to use, scenenn or scannet ")

    # files path
    parse.add_argument("--scene_num", type=str, required=True, help="which scene for mapping ")
    parse.add_argument("--result_folder", type=str,required=True,help="folder of mapping results")
    parse.add_argument("--data_folder", type=str, required=True, help="which scene for mapping ")
    parse.add_argument("--traj_filename", type=str, default="trajectory.log", help="which trajectory_to_use ")
    parse.add_argument("--gt_mask_folder", type=str, default="", help="valid when dataset is nyu gt, path to g.t. mask annotations")

    #log
    parse.add_argument("--log", type=str, default='', help="log info ")
    # mapping configuration
    parse.add_argument("--start", type=int,default=0,help="folder of mapping results")
    parse.add_argument("--end", type=int, default=-1, help="which scene for mapping ")
    parse.add_argument("--step", type=int, default=1, help="use one frame for integration every n_step frames ")
    parse.add_argument("--num_threads", type=int, default=-1, help="use one frame for integration every n_step frames ")
    parse.add_argument("--debug", action="store_true", help="whether to use visualization ")
    parse.add_argument("--preload", action="store_true", help="whether to preload images ")

    parse.add_argument("--use_temporal_results", action='store_true', help="use 2D segments results ")
    parse.add_argument("--temporal_results", action='store_true', help="save intermediate 2D segments results ")
    parse.add_argument("--temporal_results_img", action='store_true', help="save intermediate results in images ")
    parse.add_argument("--intermediate_seg_folder", type=str, default='segments', help="folder to save intermediate segments result ")

    parse.add_argument("--use_temporal_panoptics", action='store_true', help="use 2D panoptic segments ")
    parse.add_argument("--temporal_panoptics", action='store_true', help="save 2D panoptic segments ")
    parse.add_argument("--temporal_panoptics_folder", type=str, default='segments', help="folder to save 2D panoptic segments ")

    parse.add_argument("--use_temporal_geometrics", action='store_true', help="use 2D geometrics segments ")
    parse.add_argument("--temporal_geometrics", action='store_true', help="save 2D geometrics segments ")
    parse.add_argument("--temporal_geometrics_folder", type=str, default='segments', help="folder to save 2D geometrics segments result ")

    parse.add_argument("--task", type=str, default="coco80", help="coco80; nyu13; Nyu40; cocoPano")
 
    parse.add_argument("--data_association", type=int, default=0, help="0 for Ori; 1 for SemMerge; \
        2 for SemMerge+BackgroundMerge+only consider size>1000; \
        3 for SemMerge+BackgroundMerge+only consider size>1000 + considerSem when register superpoint; \
        4 for no merging; 5 for using designated superpoint id for 3D segments")
    parse.add_argument("--inst_association", type=int, default=0, help="0 for Ori; 1 for Label-Sem-Inst; \
        2 for Label-Inst-Sem; 3 for SegGraph ")


    # for SegGraph
    parse.add_argument("--seg_graph_confidence", type=int, default=0, help="0 for all confidence as 1; 1 for using inst score; \
        2 for use inst score and overlap ratio; 3 for use inst score, overlap ratio and geometric confidence")
    parse.add_argument("--use_inst_label_connect", type=int, default=1, help="")
    parse.add_argument("--connection_ratio_th", type=float, default=0.2, help="")
    parse.add_argument("--test_geometric_confidence", action='store_true', help="try test geometric confidence calculation")

    # currently not used
    parse.add_argument("--use_2D_confidence", action='store_true',help="use MaskRCNN for 2D data association")
    parse.add_argument("--geo_confidence", action='store_true',help="use geometric confidence")
    parse.add_argument("--label_confidence", action='store_true', help="use label confidence")  
    return parse.parse_args()

def ResultsDirectories(result_folder):
    result_dirs = {}

    log_file = os.path.join(result_folder, 'log')

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    result_dirs['folder'] = result_folder
    result_dirs['log'] = log_file

    return result_dirs

def main():
    import consistent_gsm
    import depth_segmentation_py

    # set files path
    args = parse_args()

    # dataset 
    dataset = args.dataset
    panoptic_node = None
    if dataset == "scenenn":
        from utils.common_scenenn import \
            Segment, SegmentsGenerator, DataLoader, isPoseValid, checkSegmentFramesEqual
        PerceptionNode = semantic_utils.PerceptionNode
        panoptic_node = PerceptionNode()
    elif dataset == "scannet":
        from utils.common_scannet import \
            Segment, SegmentsGenerator, DataLoader, isPoseValid, checkSegmentFramesEqual
        PerceptionNode = semantic_utils.PerceptionNode
        panoptic_node = PerceptionNode()
    elif dataset == "scannet_nyu":
        from utils.common_scannet_nyu import \
            Segment, SegmentsGenerator, DataLoader, isPoseValid, checkSegmentFramesEqual
        PerceptionNode = semantic_utils.Mask2FormerNode
        panoptic_node = PerceptionNode()
    elif dataset == "scannet_nyu_gt":
        from utils.common_scannet_nyu import \
            Segment, SegmentsGenerator, DataLoader, isPoseValid, checkSegmentFramesEqual
        PerceptionNode = semantic_utils.GTMaskNode
        panoptic_node = PerceptionNode(args.gt_mask_folder)
    else:
        print(" please choose suitable dataset !")
        return None

    # set configuration
    use_temporal_results = args.use_temporal_results
    save_segments = args.temporal_results
    save_results_img = args.temporal_results_img

    task = args.task
    use_geo_confidence = args.geo_confidence
    use_label_confidence = args.label_confidence
    inst_association = args.inst_association
    data_association = args.data_association

    seg_graph_confidence = args.seg_graph_confidence
    use_inst_label_connect = args.use_inst_label_connect
    connection_ratio_th = args.connection_ratio_th

    # input and output configuration
    scene_num = args.scene_num
    result_folder = args.result_folder
    data_path = args.data_folder
    scene_folder = os.path.join(data_path, scene_num)
    result_dirs = ResultsDirectories(result_folder)

    intermediate_folder = args.intermediate_seg_folder
    segments_folder = intermediate_folder

    log_info = args.log

    # DataLoader
    traj_filename = args.traj_filename
    data_loader = DataLoader(scene_folder, traj_filename, args.preload, args.preload)
    _a,depth_img,_b = data_loader.getDataFromIndex(1)
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    K_depth = data_loader.getCameraMatrix()

    # configuration
    start = args.start
    assert(start >=  data_loader.indexes[0])
    end = args.end
    if end<0:
        end = data_loader.index_max
    step = args.step
    num_threads = args.num_threads

    # initialized integrator
    log_file = os.path.abspath(result_dirs['log'])
    gsm_node = consistent_gsm.GlobalSegmentMap_py(log_file,task, use_geo_confidence,use_label_confidence,
        inst_association, data_association, num_threads, args.debug, seg_graph_confidence, 
        use_inst_label_connect==1, connection_ratio_th)
    gsm_node.outputLog(log_info)
    gsm_node.outputLog("using traj_f: "+str(data_loader.traj_f))

    if(not use_temporal_results):
        # initialized segmentors and seg generators
        dep_segmentor = depth_segmentation_py.DepthSegmentation_py(height,width,cv2.CV_32FC1, K_depth)
        segments_generator = SegmentsGenerator(gsm_node, dep_segmentor, panoptic_node, \
            save_results_img, result_dirs['folder'], save_segments, use_temporal_results, segments_folder)

        for i in tqdm(range(start,end)):
            rgb_img, depth_img, pose = data_loader.getDataFromIndex(i)
            if(rgb_img is None or depth_img is None or pose is None):
                continue
            t0 = time.time()
            segment_list = segments_generator.frameToSegments(depth_img, rgb_img, pose, i)
            gsm_node.outputLog("   Seg Generation in python cost %f s" %(time.time() - t0))

            # # check segments are saved correctly
            # depth_scaled = data_loader.getDepthScaledFromIndex(i)
            # segments_list_load = segments_generator.loadSegments(depth_scaled, K_depth, i)
            # check_equal = checkSegmentFramesEqual(segment_list, segments_list_load)
            # if(check_equal):
            #     gsm_node.outputLog("   frame %d segments checked PASS" %(i))
            # else:
            #     gsm_node.outputLog("   frame %d segments checked FAILED!!!" %(i))

            if len(segment_list) == 0:
                continue
            for segment in segment_list:
                if segment.points.shape[0] < 1:
                    continue
                if(seg_graph_confidence == 3):
                    segment.calculateBBox()
                gsm_node.insertSegments(segment.points,
                    segment.box_points, segment.instance_label,segment.class_label, 
                    segment.inst_confidence, segment.overlap_ratio ,segment.pose, segment.is_thing)
            gsm_node.integrateFrame()
            gsm_node.clearTemporaryMemory()
    else:
        # initialized seg generators
        segments_generator = SegmentsGenerator(gsm_node, None, None, \
            save_results_img, result_dirs['folder'], save_segments, use_temporal_results, segments_folder)
        for i in tqdm(range(start,end,step)):
            # read in segments
            depth_scaled = data_loader.getDepthScaledFromIndex(i)
            if depth_scaled is None:
                continue
            segment_list = segments_generator.loadSegments(depth_scaled, K_depth, i)
            if len(segment_list) == 0:
                continue
            pose_invalid = False
            for seg in segment_list:
                if not isPoseValid(seg.pose):
                    pose_invalid = True
                    break
            if pose_invalid:
                continue
            for segment in segment_list:
                if(seg_graph_confidence == 3 and segment.is_thing):
                    segment.calculateBBox()
                pose = data_loader.getPoseFromIndex(i)
                gsm_node.insertSegments(segment.points,
                    segment.box_points, segment.instance_label,segment.class_label, 
                    segment.inst_confidence, segment.overlap_ratio ,pose, segment.is_thing)
            gsm_node.integrateFrame()
            gsm_node.clearTemporaryMemory()

    # generate log and mesh
    gsm_node.LogLabelInformation()
    gsm_node.LogMeshColors()
    print("start mesh generation! ")
    gsm_node.generateMesh(result_dirs['folder'], "")
    print("finished mesh generation! ")
    

if __name__=="__main__":
    main()