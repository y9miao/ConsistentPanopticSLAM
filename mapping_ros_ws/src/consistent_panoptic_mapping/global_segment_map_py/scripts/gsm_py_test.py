# %load_ext autoreload
# %autoreload 2
import sys
vicious_path = '/home/yang/.local/lib/python2.7/site-packages'
if vicious_path in sys.path:
    sys.path.remove(vicious_path)

import depth_seg_utils
import semantic_seg_utils
from utils import *

import cv2
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import sys
sys.dont_write_bytecode = True
import os

# read images
from cv_bridge import CvBridge
import rosbag
from scipy.spatial.transform import Rotation as R

rosbag_file = "/media/yang/linux2/thesis_test/data/030/scenenn_30.bag"
rgb_topic = "/camera/rgb/image_raw"
depth_topic = "/camera/depth/image_raw"
rgb_cam_info_topic = "/camera/rgb/camera_info"
depth_cam_info_topic = "/camera/depth/camera_info"
tf_topic = "/tf"

result_file = str("./results_geo_inv2d_smallnormal")
log_file = str("./results_geo_inv2d_smallnormal/log")
maskrcnn_file = str("./results_geo_inv2d_smallnormal/maskrcnn")
depth_seg_file = str("./results_geo_inv2d_smallnormal/depth_seg")
save_intermediate_results = False
use_geo_confidence = True
use_label_confidence = False
use_instance_confidence = False

import os
if not os.path.exists(result_file):
    os.makedirs(result_file)
if not os.path.exists(log_file):
    os.makedirs(log_file)
if not os.path.exists(maskrcnn_file):
    os.makedirs(maskrcnn_file)
if not os.path.exists(depth_seg_file):
    os.makedirs(depth_seg_file)

# import python module of depth_segmentation
lib_path = "/home/yang/thesis_test/toolbox/voxbloxpp_ws/devel/lib"
sys.path.append(lib_path)
import gsm_py

bridge = CvBridge()
bag = rosbag.Bag(rosbag_file, 'r')

depth_msg_list = []
rgb_msg_list = []

depth_image_list = []
rgb_image_list = []
pose_list = []
tf_list = []
depth_cam_info = None

for topic, msg, t in bag.read_messages(topics = [depth_topic,depth_cam_info_topic,rgb_topic,tf_topic]):
    if topic == depth_cam_info_topic:
        if depth_cam_info is None:
            depth_cam_info = msg
        
    elif topic == depth_topic:
        depth_img = bridge.imgmsg_to_cv2(msg)
        depth_image_list.append(depth_img)
        depth_msg_list.append(msg)

    elif topic == rgb_topic:
        rgb_img = bridge.imgmsg_to_cv2(msg)
        rgb_image_list.append(rgb_img)
        rgb_msg_list.append(msg)

    elif topic == tf_topic:
        tf_list.append(msg)
        translation = msg.transforms[0].transform.translation
        quat = msg.transforms[0].transform.rotation
        r = R.from_quat([quat.x, quat.y, quat.z,quat.w])
        rotation = r.as_dcm()
        T_G_C = np.eye(4).astype(np.float32)
        T_G_C[:3,:3] = rotation
        T_G_C[:3,3] = [translation.x,translation.y,translation.z]
        pose_list.append(T_G_C)


# initialized segmentors
height = depth_image_list[0].shape[0]
width = depth_image_list[0].shape[1]
K_depth = np.array(depth_cam_info.K).reshape(3,3)
segmentor = depth_seg_utils.depth_segmentation_py.DepthSegmentation_py(height,width,cv2.CV_32FC1, K_depth)

node = semantic_seg_utils.MaskRCNNNode()

log_file = os.path.abspath(log_file)
gsm_node = gsm_py.GlobalSegmentMap_py(log_file,use_geo_confidence,use_label_confidence,use_instance_confidence)

# for i in range(720,920):
for i in range(0,len(depth_image_list)):
    test = None
    if(save_intermediate_results):
        test,semantic_vis,label_map = frame2Segments(segmentor, node, depth_image_list[i], \
            rgb_image_list[i],pose_list[i],save_intermediate_results)
        maskrcc_img_f = os.path.join(maskrcnn_file,str(i+1)+".png")
        depth_seg_img_f = os.path.join(depth_seg_file,str(i+1)+".png")
        cv2.imwrite(maskrcc_img_f,semantic_vis)
        cv2.imwrite(depth_seg_img_f,label_map)
    else:
        test,_,_ = frame2Segments(segmentor, node, depth_image_list[i], rgb_image_list[i], \
            pose_list[i],save_intermediate_results)

    for segment in test:
        gsm_node.insertSegments(segment.points,segment.colors, segment.geometry_confidence,
            segment.instance_label,segment.class_label, segment.label_confidence,segment.pose)

    gsm_node.integrateFrame()
    
print("finished map integration, start mesh generation! ")

result_file = os.path.abspath(result_file)
gsm_node.generateMesh(result_file)