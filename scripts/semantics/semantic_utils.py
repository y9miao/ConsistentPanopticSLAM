#!/usr/bin/env python
import sys
import time
import numpy as np
import cv2
import struct
import json
import zlib
import socket
import struct
import os
from PIL import ImageFont, ImageDraw, Image
from sensor_msgs.msg import Image, CameraInfo

DEFAULT_HOST = "0.0.0.0"
DT_SERVER_PORT = 8801
DT2_MODEL_TYPE = "Pano_seg"

class TcpClient(object):
    
    def __init__(self, ip, port):
        self.ip_ = ip
        self.port_ = port
        self.sock_ = None


    def send(self, data_bin):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect( (self.ip_, self.port_) )
        sock.sendall(data_bin)
        data_bin = self.recv_response_(sock)
        sock.close()
        return data_bin

    
    def recv_response_(self, sock):
        pack_size = sock.recv(4)
        pack_size = struct.unpack(">I", pack_size)[0]
        # fetch data package
        data_bin = self.recv_all_(sock, pack_size)
        return data_bin
    
    def recv_all_(self, sock, msg_length):
        data = b""
        size_left = msg_length

        while len(data) < msg_length:
            recv_data = sock.recv(size_left)
            size_left -= len(recv_data)
            data += recv_data

        return data

def img2bin(img):
    """
    Convert cv2 image matrix to binary string

    Args:
        img (cv2::mat): image matrix

    Returns:
        ret (binary string)
    """
    return np.array(cv2.imencode(".jpg", img)[1]).tostring() 
def bin2img(img_bin):
    """
    Convert binary string to cv2 image matrix

    Args:
        img_bin (binary string): image binary string

    Returns:
        img (cv2::mat): return opencv image matrix
    """
    return cv2.imdecode(np.frombuffer(img_bin, dtype=np.uint8), cv2.IMREAD_COLOR)
def bin2int(int_bin):
    return struct.unpack('>I', int_bin)[0]
def int2bin(integer):
    return struct.pack(">I", integer)
def pack_img(img):
    img_bin = img2bin(img)
    img_size = int2bin( len(img_bin) )

    return img_size + img_bin
def bin2mask2former_panoseg(mask2former_bin):
    """
    Unpack mask2former (Pano_seg) result data from binary

    | pkg_size (4B int) | map_size (4B int) | width (4B int) | ...
    | height (4B int) | binary_map (map_size B) | json_info_binary (rest) |
    """

    if len(mask2former_bin) == 0:
        return {}
    
    map_size = bin2int(mask2former_bin[:4])
    w, h = bin2int(mask2former_bin[4:8]), bin2int(mask2former_bin[8:12])
    seg_map_bin = zlib.decompress(mask2former_bin[12:12+map_size])
    seg_map = np.frombuffer(seg_map_bin, dtype="uint8").reshape((h, w))
    info_json = json.loads( mask2former_bin[12 + map_size:].decode() )

    mast2former_predictions = {
        "seg_map": seg_map,
        "info": info_json["info"],
    }
    return mast2former_predictions
Mask2Former_DECORDER = {
    "Pano_seg": bin2mask2former_panoseg,
}
def bin2Mask2Former(dt_bin, model_type="Pano_seg"):
    if model_type in Mask2Former_DECORDER:
        return Mask2Former_DECORDER[model_type](dt_bin)
    else:
        raise Exception("[bin2Mask2Former] Does not support model type: ".format(model_type))

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
    
class Mask2FormerNode(object):

    def __init__(
        self,
        detectron_ip = DEFAULT_HOST,
        detectron_port = DT_SERVER_PORT,
    ):
        self.dt_client_ = TcpClient(ip=detectron_ip, port=detectron_port)
        self.use_gt = False

    def forward(self, rgb_img):
        start = time.time()
        img_bin = pack_img(rgb_img)
        resp_bin = self.dt_client_.send(img_bin)
        resp = bin2Mask2Former(resp_bin)
        end = time.time() 
        # print("time_used: %f"%(end-start))

        return resp

PALETTE = [
    [120, 120, 120], # for background
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8]
]
SCANNET_SEM_SEG_CATEGORIES = {
    1: {'name':"wall", 'isthing':False, 'color': PALETTE[1]},
    2: {'name':"floor", 'isthing':False, 'color': PALETTE[2]},
    3: {'name':"cabinet", 'isthing':True, 'color': PALETTE[3]},
    4: {'name':"bed", 'isthing':True, 'color': PALETTE[4]},
    5: {'name':"chair", 'isthing':True, 'color': PALETTE[5]},
    6: {'name':"sofa", 'isthing':True, 'color': PALETTE[6]},
    7: {'name':"table", 'isthing':True, 'color': PALETTE[7]},
    8: {'name':"door", 'isthing':True, 'color': PALETTE[8]},
    9: {'name':"window",'isthing':True, 'color': PALETTE[9]},
    10: {'name':"bookshelf",'isthing':True, 'color': PALETTE[10]},
    11: {'name':"picture",'isthing':True, 'color': PALETTE[11]},
    12: {'name':"counter",'isthing':True, 'color': PALETTE[12]},
    13: {'name':"blinds",'isthing':False, 'color': PALETTE[13]},
    14: {'name':"desk",'isthing':True, 'color': PALETTE[14]},
    15: {'name':"shelves",'isthing':False, 'color': PALETTE[15]},
    16: {'name':"curtain",'isthing':True, 'color': PALETTE[16]},
    17: {'name':"dresser",'isthing':False, 'color': PALETTE[17]},
    18: {'name':"pillow",'isthing':True, 'color': PALETTE[18]},
    19: {'name':"mirror",'isthing':False, 'color': PALETTE[19]},
    20: {'name':"floor mat",'isthing':False, 'color': PALETTE[20]},
    21: {'name':"clothes",'isthing':False, 'color': PALETTE[21]},
    22: {'name':"ceiling",'isthing':False, 'color': PALETTE[22]},
    23: {'name':"books",'isthing':False, 'color': PALETTE[23]},
    24: {'name':"refridgerator",'isthing':True, 'color': PALETTE[24]},
    25: {'name':"television",'isthing':True, 'color': PALETTE[25]},
    26: {'name':"paper",'isthing':False, 'color': PALETTE[26]},
    27: {'name':"towel",'isthing':False, 'color': PALETTE[27]},
    28: {'name':"shower curtain",'isthing':True, 'color': PALETTE[28]},
    29: {'name':"box",'isthing':False, 'color': PALETTE[29]},
    30: {'name':"whiteboard",'isthing':False, 'color': PALETTE[30]},
    31: {'name':"person",'isthing':True, 'color': PALETTE[31]},
    32: {'name':"nightstand",'isthing':True, 'color': PALETTE[32]},
    33: {'name':"toilet",'isthing':True, 'color': PALETTE[33]},
    34: {'name':"sink",'isthing':True, 'color': PALETTE[34]},
    35: {'name':"lamp",'isthing':True, 'color': PALETTE[35]},
    36: {'name':"bathtub",'isthing':True, 'color': PALETTE[36]},
    37: {'name':"bag",'isthing':True, 'color': PALETTE[37]},
    38: {'name':"otherstructure",'isthing':False, 'color': PALETTE[38]},
    39: {'name':"otherfurniture",'isthing':True, 'color': PALETTE[39]},
    40: {'name':"otherprop",'isthing':False, 'color': PALETTE[40]},
}

def getMetaData():
    meta = {}
    thing_classes = []
    thing_colors = []
    stuff_classes = []
    stuff_colors = []
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    contiguous_id_to_stuff_dataset_id = {}
    cate_ids = list(SCANNET_SEM_SEG_CATEGORIES.keys())
    cate_ids.sort()
    for cate_id in cate_ids:
        cate_item = SCANNET_SEM_SEG_CATEGORIES[cate_id]
        # if cate_item['isthing']:
        thing_classes.append(cate_item['name'])
        thing_colors.append(cate_item['color'])
        # in order to use sem_seg evaluator
        stuff_classes.append(cate_item['name'])
        stuff_colors.append(cate_item['color'])

    for i, cat_id in enumerate(cate_ids):
        cat = SCANNET_SEM_SEG_CATEGORIES[cat_id]
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat_id] = i
        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat_id] = i
        contiguous_id_to_stuff_dataset_id[i] = cat_id

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["contiguous_id_to_stuff_dataset_id"] = contiguous_id_to_stuff_dataset_id

    meta["categories"] = SCANNET_SEM_SEG_CATEGORIES

    return meta

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

class GTMaskNode():
    def __init__(self, gt_folder, use_contiguous_cate = True):
        self.use_gt = True

        self.seq_num = os.path.basename(gt_folder)

        self.meta = getMetaData()
        # load gt semantic masks
        annotation_file = os.path.join(gt_folder, "annotations.json")
        annotations = None
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        img_size = annotations['img_size']

        frames_item = {}
        for frame_info in annotations['frames_info']:
            segments_info = frame_info['segments_info']
            # map cate_id in segments_info to contiguous cate ids
            if use_contiguous_cate:
                for segment_info in segments_info:
                    if segment_info["category_id"] in self.meta["thing_dataset_id_to_contiguous_id"]:
                        segment_info["category_id"] = self.meta["thing_dataset_id_to_contiguous_id"][
                            segment_info["category_id"] ]
                    else:
                        segment_info["category_id"] = self.meta["stuff_dataset_id_to_contiguous_id"][
                            segment_info["category_id"] ]
            frame_item = {
                'file_name':frame_info['file_name'], 
                'image_id': self.seq_num + '.' + frame_info['image_id'].split('.')[-1],
                'height': img_size[0],
                'width': img_size[1],
                'pan_seg_file_name': frame_info['pan_seg_file_name'],
                'segments_info': segments_info
            }
            frames_item[frame_item['image_id']] = frame_item
        self.frames_item = frames_item
        pass

    def forward(self, frame_idx):
        image_id = self.seq_num + "." + str(frame_idx)
        assert( image_id in self.frames_item)
        frame_item = self.frames_item[image_id]

        from PIL import Image
        panoptic_mask_color = Image.open(frame_item['pan_seg_file_name'])
        panoptic_mask_color_arr = np.array(panoptic_mask_color)
        panoptic_mask_id_arr = rgb2id(panoptic_mask_color_arr)
        ids_unique = np.unique(panoptic_mask_id_arr)
        ids_to_contiguous_ids = {}
        # get panoptic mask
        id_contiguous = 1
        ids_to_segs_area = {}
        for id in ids_unique:
            if id == 0:
                continue
            ids_to_contiguous_ids[id] = id_contiguous
            seg_mask = (panoptic_mask_id_arr==id)
            panoptic_mask_id_arr[seg_mask] = id_contiguous
            ids_to_segs_area[id] = np.sum(seg_mask)
            id_contiguous += 1
        # get segments information
        id_to_segs_info = {}
        for seg_info in frame_item['segments_info']:
            id_to_segs_info[seg_info['id']] = seg_info

        segs_info_out = []
        for id in ids_to_contiguous_ids:
            seg_info = id_to_segs_info[id]
            seg_info_out = {
                'id': ids_to_contiguous_ids[id],
                'isthing': None,
                'category_id': int(seg_info['category_id']),
                'area': int(ids_to_segs_area[id]),
            }
            # determine isThing
            cate_id_dataset = self.meta["contiguous_id_to_stuff_dataset_id"][seg_info['category_id']]
            seg_info_out['isthing'] = SCANNET_SEM_SEG_CATEGORIES[cate_id_dataset]['isthing']
            segs_info_out.append(seg_info_out)

        gt_predictions = {
            'seg_map': panoptic_mask_id_arr.astype(np.uint8),
            'info': segs_info_out
        }
        return gt_predictions
