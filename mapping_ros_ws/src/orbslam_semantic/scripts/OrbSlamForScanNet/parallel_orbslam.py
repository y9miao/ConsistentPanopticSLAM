import os
import io
import argparse
import numpy as np
import copy
import pickle
import yaml
import cv2
import subprocess
import threading
import queue

class BashThread(threading.Thread):
    def __init__(self,task_queue, id):
        threading.Thread.__init__(self)
        self.queue = task_queue
        self.th_id = id
        self.start()

    def run(self):
        while True:
            try:
                command = self.queue.get(block=False)
                # out_print = "thread %d is processing %d"%(self.th_id, command)
                # os.system(" echo %s"%(out_print))
                # os.system(command)
                subprocess.call(command, shell=True)
                self.queue.task_done()
            except queue.Empty:
                break

class BashThreadPool():
    def __init__(self,task_queue,thread_num):
        self.queue = task_queue
        self.pool = []
        for i in range(thread_num):
            self.pool.append(BashThread(task_queue, i))

    def joinAll(self):
        self.queue.join()

def RunBashBatch(commands, jobs_per_step = 1):
    # task submission
    commands_queue = queue.Queue()
    for command in commands:
        commands_queue.put(command)
    map_eval_thread_pool = BashThreadPool(commands_queue, jobs_per_step)
    map_eval_thread_pool.joinAll()

def writeDictToYaml(dict_yaml, config_file):
    with open(config_file, 'w') as f:
        f.write('%YAML:1.0\n') # yaml1.0 head
        for item in dict_yaml:
            if isinstance(dict_yaml[item], str):
                f.write(str(item) + ': "' + str(dict_yaml[item]) + '"\n')
            else:
                f.write(str(item) + ': ' + str(dict_yaml[item]) + '\n')
    return None

def prepareSettingFile( scene_folder, orbslam_config_template_file):

    # <orblsam config>
    # Read YAML file 
    orbslam_config_template = None
    with open(orbslam_config_template_file, 'r') as yaml_stream:
        orbslam_config_template = yaml.safe_load(yaml_stream)

    # generate orbslam config file
    # write seq-specific configs
    orbslam_config = copy.deepcopy(orbslam_config_template)

    depth_intrinsic_file = os.path.join(scene_folder, 'intrinsic', 'intrinsic_depth.txt')
    intrinsics = np.loadtxt(depth_intrinsic_file)
    orbslam_config['Camera1.fx'] = float(intrinsics[0,0])
    orbslam_config['Camera1.fy'] = float(intrinsics[1,1])
    orbslam_config['Camera1.cx'] = float(intrinsics[0,2])
    orbslam_config['Camera1.cy'] = float(intrinsics[1,2])

    # save config file
    dest_folder = os.path.join(scene_folder, "orbslam")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_config_file = os.path.join(dest_folder, "orbslam.yaml")
    writeDictToYaml(orbslam_config, dest_config_file)

def main():

    DataFolder = "/home/yang/990Pro/scannet_seqs/data/val"
    seqs_names = os.listdir(DataFolder)

    orbslam_config_template = "/home/yang/toolbox/test_field/volumetric-semantically-consistent-3D-panoptic-mapping/"+ \
        "mapping_ros_ws/src/orbslam_semantic/examples/orbslam_python.yaml"
    
    # generate config files
    # for seq in seqs_names:
    #     seq_folder = os.path.join(DataFolder, seq)
    #     prepareSettingFile( seq_folder, orbslam_config_template )
        
    seq_selected = []
    for seq in seqs_names:
        est_traj_f = os.path.join(DataFolder, seq, "orbslam", "traj_orbslam.log")
        if not os.path.isfile(est_traj_f):
            seq_selected.append(seq)
    # execute mapping 
    exe_file = "/home/yang/toolbox/test_field/volumetric-semantically-consistent-3D-panoptic-mapping/mapping_ros_ws/src/orbslam_semantic/scripts/OrbSlamForScanNet/orbslam_scannet.sh"
    exe_commands = []
    
    for seq in seqs_names:
        seq_folder = os.path.join(DataFolder, seq)
        orbslam_setting_f = os.path.join(seq_folder, "orbslam", "orbslam.yaml")

        depth_folder = os.path.join(seq_folder, "depth")
        color_folder = os.path.join(seq_folder, "color_warped")
        rgb_num = len(os.listdir(color_folder))
        depth_num = len(os.listdir(depth_folder))
        img_num = min(rgb_num, depth_num)

        args = seq_folder + " " + orbslam_setting_f + " " + str(img_num)
        exe_commands.append("bash " + exe_file + " " + args) 

    RunBashBatch(exe_commands, 10)

if __name__ == '__main__':
    main()
    