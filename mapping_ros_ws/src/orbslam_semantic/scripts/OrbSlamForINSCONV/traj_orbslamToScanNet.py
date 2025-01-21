import numpy as np
import os 
import argparse
from scipy.spatial.transform import Rotation as R

def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # files path
    parse.add_argument("--traj_orbslam", type=str, required=True, help=" input")
    parse.add_argument("--scene_folder", type=str,required=True,help=" output")
    
    return parse.parse_args()

def logToTraj(log_file):
    trajectory = {}
    f = open(log_file,'r')
    T_WC = []
    current_id = None
    for line in f.readlines():
        data = line.split(' ')
        if(len(data) == 3):
            if T_WC:
                T_WC = np.array(T_WC)
                r = R.from_matrix(T_WC[:3,:3])
                T_WC[:3,:3] = r.as_matrix() # ensure rotation matrix to be valid
                trajectory[current_id] = np.array(T_WC).reshape(4,4)
            current_id = int(data[0])
            T_WC = []
        elif(len(data) == 4):
            T_WC.append([float(data[0]),float(data[1]),float(data[2]),float(data[3])])
    f.close()
    return trajectory

def main():
    args = parse_args()
    traj_orb_slam_f = args.traj_orbslam
    
    scene_folder = args.scene_folder
    
    traj_orbslam_arr = np.loadtxt(traj_orb_slam_f)
    frame_ids = traj_orbslam_arr[:,0].astype(np.int32)
    unique_frame_ids, unique_frame_ids_indexs = np.unique(frame_ids, return_index=True)
    # repetitive_frame_ids = unique_frame_ids[counts>1]
    num_est_poses = unique_frame_ids.shape[0]

    traj_orb = {}
    for frame_i in unique_frame_ids_indexs:

        frame_id = int(traj_orbslam_arr[frame_i,0])
        t_WC = traj_orbslam_arr[frame_i, 1:4]
        q_WC = traj_orbslam_arr[frame_i, 4:8]
        T_WC = np.identity(4)

        r = R.from_quat(q_WC)
        T_WC[:3,:3] = r.as_matrix()
        T_WC[:3,3] = t_WC
        traj_orb[frame_id] = T_WC
        
    out_traj_f = os.path.join(scene_folder, "orbslam", "traj_orbslam.log")

    # write results
    with open(out_traj_f, 'w') as f:
        for frame_id in traj_orb:
            JOIN_ = " "
            frame_id_log = JOIN_.join([str(int(frame_id)), \
                                str(int(frame_id)), str(int(frame_id+1))]) + '\n'
            f.write(frame_id_log)

            T_WC = traj_orb[frame_id]
            for matrix_line_i in range(T_WC.shape[0]):
                matrix_line = list(T_WC[matrix_line_i])
                matrix_line_str = [str(item) for item in matrix_line]
                matrix_line_log = JOIN_.join(matrix_line_str) + '\n'
                f.write(matrix_line_log)

if __name__=="__main__":
    main()

    breakpoint = None