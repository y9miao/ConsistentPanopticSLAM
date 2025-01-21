import pickle
import os 
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

def TrajLogToTraj(traj_file):
    trajectory = {}
    f = open(traj_file,'r')
    T_WC = []
    current_id = None
    for line in f.readlines():
        data = line.split(' ')
        if(len(data) == 3):
            if T_WC:
                T_WC = np.array(T_WC)
                r = Rotation.from_matrix(T_WC[:3,:3])
                T_WC[:3,:3] = r.as_matrix() # ensure rotation matrix to be valid
                trajectory[current_id] = np.array(T_WC).reshape(4,4)
            current_id = int(data[0])
            T_WC = []
        elif(len(data) == 4):
            T_WC.append([float(data[0]),float(data[1]),float(data[2]),float(data[3])])
    f.close()
    return trajectory

def TrajTxtToTraj(traj_file, fps=15):
    trajectory = {}
    
    traj_arr = np.loadtxt(traj_file)
    time_stamps = traj_arr[:,0]
    frame_ids = np.rint(time_stamps * fps)
    unique_frame_ids, unique_frame_ids_indexs = np.unique(frame_ids, return_index=True)
    
    for frame_i in unique_frame_ids_indexs:
        frame_id = frame_ids[frame_i]
        t_WC = traj_arr[frame_i, 1:4]
        q_WC = traj_arr[frame_i, 4:8]
        T_WC = np.identity(4)
        r = Rotation.from_quat(q_WC)
        T_WC[:3,:3] = r.as_matrix()
        T_WC[:3,3] = t_WC
        trajectory[frame_id] = T_WC
    return trajectory

def TrajToTxt(traj, txt_file, fps = 15.0):
    f = open(txt_file,'w')
    frame_idx_list = list(traj.keys())
    frame_idx_list.sort()
    for frame_idx in frame_idx_list:
        timestamp = frame_idx * 1.0 / fps

        T_WC = traj[frame_idx]

        r = Rotation.from_matrix(T_WC[:3,:3])
        quat_wc = r.as_quat()
        translation = T_WC[:3, 3]
        translation_str = " ".join([str(item) for item in translation])
        quat_wc_str = " ".join([str(item) for item in quat_wc])
        pose_line = str(timestamp) + " " + translation_str + " " + quat_wc_str
        f.write(pose_line+"\n")
    f.close()

def generateSeqEvalFiles(
        gt_traj_log, 
        est_traj_txt,
        source_cfg_file,
        target_folder):

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # transform and save traj
    target_gt_file = os.path.join(target_folder, "stamped_groundtruth.txt")
    target_est_file = os.path.join(target_folder, "stamped_traj_estimate.txt")

    gt_traj = TrajLogToTraj(gt_traj_log)
    est_traj = TrajTxtToTraj(est_traj_txt)
    TrajToTxt(gt_traj, target_gt_file)
    TrajToTxt(est_traj, target_est_file)

    # save eval cfg file
    command = "cp " + source_cfg_file + " " + os.path.join(target_folder, "eval_cfg.yaml")
    subprocess.call(command, shell=True)

def transformOrbslamTrajForEval(DataFolder, ResultFolder, Seqs, 
        EvalFolder, eval_name, source_cfg_file, platform = 'desktop'):
    
    for seq in Seqs:
        gt_traj_log = os.path.join(DataFolder, seq, "trajectory.log")
        est_traj_txt = os.path.join(ResultFolder, seq, "traj", "trajectory_orb_slam.txt")

        seq_eval_folder_name = platform + "_" + eval_name + "_" + ''.join(seq.split('_'))
        target_folder = os.path.join(EvalFolder, platform, eval_name, seq_eval_folder_name)

        generateSeqEvalFiles(gt_traj_log, est_traj_txt, source_cfg_file, target_folder)


DataFolder = "/home/yang/990Pro/scannet_seqs/data"
Seqs = os.listdir(DataFolder)
EvalFolder = "/home/yang/toolbox/test_field/rpg_trajectory_evaluation/results/scannet"
source_cfg_file =  "/home/yang/toolbox/test_field/rpg_trajectory_evaluation/results/voxgraph/eval_cfg.yaml"

# # orignal orbslam
# ResultFolder = "/home/yang/990Pro/scannet_seqs/semantic_orbslam/original_orbslam"
# eval_name = "OrbslamSemOriginal"
# transformOrbslamTrajForEval(DataFolder, ResultFolder, Seqs, 
#     EvalFolder, eval_name, source_cfg_file)

# # orbslam with semantics - feature filtering
# ResultFolder = "/home/yang/990Pro/scannet_seqs/semantic_orbslam/orbslam_feature_filter"
# eval_name = "OrbslamSemFeatureFilter"
# transformOrbslamTrajForEval(DataFolder, ResultFolder, Seqs, 
#     EvalFolder, eval_name, source_cfg_file)

# orbslam with semantics - feature filtering
ResultFolder = "/home/yang/990Pro/scannet_seqs/semantic_orbslam/orbslam_feature_seg_filter"
eval_name = "OrbslamSegFeatureFilter"
transformOrbslamTrajForEval(DataFolder, ResultFolder, Seqs, 
    EvalFolder, eval_name, source_cfg_file)