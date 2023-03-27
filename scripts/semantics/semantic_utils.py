#!/usr/bin/env python
import sys
import time
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from sensor_msgs.msg import Image, CameraInfo
from panoptic_mapping.utils.semantics.tcp_client import TcpClient
from panoptic_mapping.utils.semantics.tcp_utils import pack_img, bin2detectron

DEFAULT_HOST = "0.0.0.0"
DT_SERVER_PORT = 8801
DT2_MODEL_TYPE = "Pano_seg"

class PerceptionNode(object):

    def __init__(
        self,
        detectron_ip = DEFAULT_HOST,
        detectron_port = DT_SERVER_PORT,
        detectron_model = DT2_MODEL_TYPE
    ):
        
        self.dt_client_ = TcpClient(ip=detectron_ip, port=detectron_port)
        self.dt_model_ = detectron_model
        self.visualize_count = 0


    def __def__(self):
        pass

    def forward(self, rgb_img):
        start = time.time()
        img_bin = pack_img(rgb_img)
        resp_bin = self.dt_client_.send(img_bin)
        resp = bin2detectron(resp_bin, self.dt_model_)
        end = time.time() 
        # print("time_used: %f"%(end-start))
        return resp
                                            
    def visualize(self, resp):
        # visualize
        from scipy import stats as st
        from panoptic_mapping.utils.semantics.pano_colormap import color_map, cate_id_info
        from scipy import stats as st
        # text setting
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1.3
        fontColor              = (255,255,255)
        thickness              = 4
        lineType               = 2

        # decode panoptic result
        seg_map = resp["seg_map"]
        info  = resp["info"]
        # boxes = resp["boxes"]
        # boxes = np.reshape(boxes, len(boxes)*4)
        obj_id = []
        sem_id = []
        scores = []
        area = []
        obj_category = []
        sem_category = []

        for i in range(len(info)):
            info_current = info[i]
            if info_current["isthing"]:
                scores.append(info_current["score"])
                obj_category.append(info_current["category_id"])
                obj_id.append(info_current["id"])  
            else:
                area.append(info_current["area"])
                sem_category.append(info_current["category_id"]+80) # start from 81, no class 80
                sem_id.append(info_current["id"])

        height, width = resp["seg_map"].shape[0], resp["seg_map"].shape[1]
        color_image = np.zeros((height, width, 3)).astype(np.uint8)
        # seg_map = seg_map.reshape(height, width)
        ids = np.unique(color_image)
        color_image[seg_map==0] = [200,200,200]
        for obj_index in range(len(obj_category)):
            id = obj_id[obj_index]
            category = obj_category[obj_index]
            color_image[seg_map==id] = color_map[category]
            # print("category = %d "%(category))
            # print(" color: ", color_map[category])
        for semantic_index in range(len(sem_category)):
            id = sem_id[semantic_index]
            category = sem_category[semantic_index]
            color_image[seg_map==id] = color_map[category] 

        pass_mergeed_table = False
        for obj_index in range(len(obj_category)):
            # box = boxes[obj_index]
            # x_center = np.ceil(box[0]+box[2])/2
            # y_center = np.ceil(box[1]+box[3])/2
            id = obj_id[obj_index]
            x_pos, y_pos = np.where(seg_map==id)
            x_center = np.mean(x_pos)
            y_center = np.mean(y_pos)
            bottomLeftCornerOfText = (int(y_center), int(x_center))
            category = obj_category[obj_index]
            # color_image = cv2.putText(color_image, "inst=" + cate_id_info[category]['name'],
            #     bottomLeftCornerOfText, font, fontScale,
                # fontColor,thickness,lineType)
            name = cate_id_info[category]['name'].split('-')[0]
            
            color_image = cv2.putText(color_image, name.capitalize(),
                bottomLeftCornerOfText, font, fontScale,
                fontColor,thickness,lineType)

        for semantic_index in range(len(sem_category)):
            id = sem_id[semantic_index]
            category = sem_category[semantic_index]
            x_pos, y_pos = np.where(seg_map==id)
            x_center = np.mean(x_pos)
            y_center = np.mean(y_pos)
            
            # color_image = cv2.putText(color_image, "sem=" + cate_id_info[category+20]['name'],
            #     bottomLeftCornerOfText, font, fontScale,
            #     fontColor,thickness,lineType)
            name = cate_id_info[category+20]['name'].split('-')[0]
            bottomLeftCornerOfText = (int(y_center), int(x_center))
            color_image = cv2.putText(color_image, name.capitalize(),
                bottomLeftCornerOfText, font, fontScale,
                fontColor,thickness,lineType)
        return color_image