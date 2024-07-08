# this script is to read in RGB-D frames and then to apply consistent semantic graph cut and instance segmentation refinement given 

import sys
import os
import argparse
import numpy as np
from panoptic_mapping.utils.segment_graph.utils import *
from panoptic_mapping.utils.segment_graph import semantic_graph_cut
from panoptic_mapping.utils.segment_graph import seg_graph_tune
import open3d.core as o3c
import time

cur_file_path = os.path.dirname(os.path.abspath(__file__))
cur_parent_path = os.path.dirname(cur_file_path)
sys.path.insert(0, cur_parent_path)

def parse_args():
    data_path = os.environ.get('DATA')
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # files path
    parse.add_argument("--scene_num", type=str, required=True, help="which scene for mapping ")
    parse.add_argument("--result_folder", type=str,required=True,help="folder of mapping results")
    parse.add_argument("--base_folder", type=str,required=True,help="folder of initial segment graph")
    #log
    parse.add_argument("--log", type=str, default='', help="log info ")
    #configure
    parse.add_argument("--task", type=str, default='CoCoPano', help="task CoCoPano ")
    # parameter for instnace segmentation refinement
    parse.add_argument("--break_weak", type=int, default=1, help="hyper-parameter for Binary Energy ")
    parse.add_argument("--break_params", type=str, default="0.1_0.7_0.05_0.7", help="hyper-parameter for Break Reunion")
    # parameter for semantic regularization
    parse.add_argument("--K", type=float, default=15.0, help="hyper-parameter for Binary Energy ")
    parse.add_argument("--theta", type=float, default=0.5, help="hyper-parameter for Binary Energy ")

    return parse.parse_args()

def ResultsDirectories(result_folder):
    result_dirs = {}

    log_folder = os.path.join(result_folder, 'log')

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    result_dirs['folder'] = result_folder
    result_dirs['log'] = log_folder

    return result_dirs

def main():
    # set files path
    args = parse_args()

    # task
    BackgroundSemLabel = 80
    task = args.task
    if(task == "CoCoPano"):
        BackgroundSemLabel = 80
    # input and output configuration
    scene_num = args.scene_num
    base_folder = args.base_folder
    result_dirs = ResultsDirectories(args.result_folder)
    result_folder = result_dirs['folder']
    log_folder = result_dirs['log']

    # input files
    confidence_file = os.path.join(base_folder, 'log', "ConfidenceMap.txt")
    initial_guess_file = os.path.join(base_folder, 'log', "LabelInitialGuess.txt")
    label_mesh_f = os.path.join(base_folder, 'geo_sem_eval', "label_mesh_proj.ply")

    # output files 
    log_sem_f = os.path.join(log_folder, 'log_semantic_graph_cut.txt')
    log_inst_f = os.path.join(log_folder, 'log_inst_refine.txt')
    eval_folder = os.path.join(result_folder, 'geo_sem_eval')
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    out_semantic_mesh_f = os.path.join(eval_folder, 'semantic_mesh_proj.ply')
    out_inst_mesh_f = os.path.join(eval_folder, 'instance_mesh_proj.ply')
    log_info = args.log + '\n'

    # hyper-parameter
    break_weak = bool(args.break_weak)
    break_params = args.break_params
    break_params = [float(param) for param in break_params.split("_")]
    K = args.K
    theta = args.theta

    # device      
    device = o3c.Device("CPU", 0)
    if(o3c.cuda.is_available()):
        device= o3c.Device("CUDA", 0)

    # load initial info
    labels_info, instances_info, semantic_instance_map, confidence_map = \
        loadLabelInitualGuess(initial_guess_file, confidence_file)

    # log io
    log_inst_io = open(log_inst_f, 'w')
    log_sem_io = open(log_sem_f, 'w')

    # graph-cut for semantic regularization
    
    semantic_graphcut = semantic_graph_cut.SemanticGraphCut(instances_info, labels_info, confidence_map, device, \
        log_sem_io, K=K, theta=theta)
    semantic_graphcut.log_to_file(log_info)
    time_sem_start = time.time()
    instances_info_regularized, labels_info_regularized, sem_regularied_segs = semantic_graphcut.regularizeSemantic()
    time_sem_finished = time.time()

    # instances_info_regularized, labels_info_regularized, sem_regularied_segs = \
    #     instances_info, labels_info, []

    # break reunion for inst refinement
    
    
    inst_seg_graph = seg_graph_tune.SegGraph(instances_info_regularized, labels_info_regularized, confidence_map, \
        semantic_instance_map, semantic_updated_segs=sem_regularied_segs, \
        log_io=log_inst_io, device=device, break_weak_connection=break_weak, \
        parameter=break_params)
    time_inst_start = time.time()
    instances_info_refined, labels_info_refined, labels_updated = inst_seg_graph.refineLIMC3BreakAllReunion()
    time_inst_finished = time.time()
    # out put mesh
    semantic_graphcut.generateSemanticMesh(labels_info_refined, label_mesh_f, out_semantic_mesh_f)
    inst_seg_graph.generateInstanceMesh(instances_info_refined, labels_updated, label_mesh_f,  out_inst_mesh_f)

    log_sem_io.write("\n semantic regularization cost "+ str(time_sem_finished-time_sem_start) + " seconds\n")
    log_inst_io.write("\n inst refinement cost "+ str(time_inst_finished-time_inst_start) + " seconds\n")

    log_inst_io.close()
    log_sem_io.close()
if __name__=="__main__":
    main()