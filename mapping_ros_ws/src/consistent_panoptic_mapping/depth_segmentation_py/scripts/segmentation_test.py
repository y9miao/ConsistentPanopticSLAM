import sys
import os
import rospy
import rospkg
import cv2
import numpy as np

# import python module of depth_segmentation
lib_path = "/home/yang/thesis_test/toolbox/voxbloxpp_ws/devel/lib"
sys.path.append(lib_path)
import depth_segmentation_py
segmentor_test = depth_segmentation_py.DepthSegmentation_py(10,10,cv2.CV_32FC1, np.eye(3))
depth_image_np = np.ones((10,10)).astype("float32")
depth_map_np = np.ones((10,10,3)).astype("float32")
segmentor_test.computeDepthMap_py( depth_image_np, depth_map_np)