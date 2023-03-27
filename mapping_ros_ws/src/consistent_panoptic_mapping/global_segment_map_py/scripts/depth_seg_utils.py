import sys
import cv2
import numpy as np

# import python module of depth_segmentation
lib_path = "/home/yang/thesis_test/toolbox/voxbloxpp_ws/devel/lib"
sys.path.append(lib_path)

import depth_segmentation_py

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

def show_images(images_list, mode = 'column', cmap = None, size = (5,4)):
    if(len(images_list)>10):
        print(" too much images!")
        return None

    num_images = len(images_list)
    import matplotlib.pyplot as plt
    plt.figure(figsize=size)
    # fig.set_figheight(15)
    # fig.set_figwidth(15)
    if mode == 'column':
        for index,image in enumerate(images_list):
            plt.subplot(num_images,1,index+1)
            plt.imshow(image, cmap)
    elif mode == 'row':
        for index,image in enumerate(images_list):
            plt.subplot(1,num_images,index+1)
            plt.imshow(image, cmap)

# geometric confidence calculation 

# given normal map and depth map, output (1+cos(theta))/2
def normal_confidence(normal_map, depth_image, depth_map):
    cos_angle_map = -1*np.ones_like(depth_image)
    valid_norms = ~np.isnan(normal_map[:,:,0])
    inner_product = np.sum(np.multiply(normal_map[valid_norms],depth_map[valid_norms]), axis=-1)
    inner_product_normalized = inner_product/(np.linalg.norm(normal_map[valid_norms],axis=-1)*np.linalg.norm(depth_map[valid_norms],axis=-1))
    cos_angle_map[valid_norms] = -inner_product_normalized # reverse because the normals points at camera
    cos_angle_map_normalized = 0.02*(cos_angle_map+1)/2
    return cos_angle_map_normalized

# output inverse depth values(meter^-1)
def depth_confidence(depth_image, thresholds = [0.1, 10]):
    depth_confidence_map = np.zeros_like(depth_image)
    valid_depth = np.logical_and(depth_image>0.1, depth_image<10)
    depth_confidence_map[valid_depth] = 1/(depth_image[valid_depth]**2)
    return depth_confidence_map

def confidence_calculation(depth_img_scaled, depth_map, normal_map):
    cos_angle_map = normal_confidence(normal_map, depth_img_scaled, depth_map)
    depth_confidence_map = depth_confidence(depth_img_scaled)
    return cos_angle_map, depth_confidence_map

# calculate geometry confidence given normal_confidence and depth_confidence
def geometry_confidence_calculation(cos_angle_map, depth_confidence_map):
    geometry_confidence_map = np.zeros_like(cos_angle_map)
    valid = np.logical_and((cos_angle_map>1e-3),depth_confidence_map>1e-3)
    geometry_confidence_map[valid] =  cos_angle_map[valid]+1.0*depth_confidence_map[valid]
    return geometry_confidence_map
