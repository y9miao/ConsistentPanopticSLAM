import numpy as np
color_map = [[220, 20, 60],   [119, 11, 32],   [0, 0, 142],     [0, 0, 230],     [106, 0, 228], 
           [0, 60, 100],    [0, 80, 100],    [0, 0, 70],      [0, 0, 192],     [250, 170, 30], 
           [100, 170, 30],  [220, 220, 0],   [175, 116, 175], [250, 0, 30],    [165, 42, 42], 
           [255, 77, 255],  [0, 226, 252],   [182, 182, 255], [0, 82, 0],      [120, 166, 157], 
           [110, 76, 0],    [174, 57, 255],  [199, 100, 0],   [72, 0, 118],    [255, 179, 240], 
           [0, 125, 92],    [209, 0, 151],   [188, 208, 182], [0, 220, 176],   [255, 99, 164], 
           [92, 0, 73],     [133, 129, 255], [78, 180, 255],  [0, 228, 0],     [174, 255, 243], 
           [45, 89, 255],   [134, 134, 103], [145, 148, 174], [255, 208, 186], [197, 226, 255], 
           [171, 134, 1],   [109, 63, 54],   [207, 138, 255], [151, 0, 95],    [9, 80, 61], 
           [84, 105, 51],   [74, 65, 105],   [166, 196, 102], [208, 195, 210], [255, 109, 65], 
           [0, 143, 149],   [179, 0, 194],   [209, 99, 106],  [5, 121, 0],     [227, 255, 205], 
           [147, 186, 208], [153, 69, 1],    [3, 95, 161],    [163, 255, 0],   [119, 0, 170], 
           [0, 182, 199],   [0, 165, 120],   [183, 130, 88],  [95, 32, 0],     [130, 114, 135], 
           [110, 129, 133], [166, 74, 118],  [219, 142, 185], [79, 210, 114],  [178, 90, 62], 
           [65, 70, 15],    [127, 167, 115], [59, 105, 106],  [142, 108, 45],  [196, 172, 0], 
           [95, 54, 80],    [128, 76, 255],  [201, 57, 1],    [246, 0, 122],   [191, 162, 208], 
           [200,200,200],
           [255, 255, 128], [147, 211, 203], [150, 100, 100], [168, 171, 172], [146, 112, 198],
           [210, 170, 100], [92, 136, 89],   [218, 88, 184],  [241, 129, 0],   [217, 17, 255], 
           [124, 74, 181],  [70, 70, 70],    [255, 228, 255], [154, 208, 0],   [193, 0, 92], 
           [76, 91, 113],   [255, 180, 195], [106, 154, 176], [230, 150, 140], [60, 143, 255], 
           [128, 64, 128],  [92, 82, 55],    [254, 212, 124], [73, 77, 174],   [255, 160, 98], 
           [255, 255, 255], [104, 84, 109],  [169, 164, 131], [225, 199, 255], [137, 54, 74], 
           [135, 158, 223], [7, 246, 231],   [107, 255, 200], [58, 41, 149],   [183, 121, 142], 
           [255, 73, 97],   [107, 142, 35],  [190, 153, 153], [146, 139, 141], [70, 130, 180], 
           [134, 199, 156], [209, 226, 140], [96, 36, 108],   [96, 96, 96],    [64, 170, 64], 
           [152, 251, 152], [208, 229, 228], [206, 186, 171], [152, 161, 64],  [116, 112, 0], 
           [0, 114, 143],   [102, 102, 156], [250, 141, 255]]

import pickle
cate_id_info= None
cate_id_info_f = "/home/yang/toolbox/scripts/panoptic_mapping/utils/semantics/cate_id_info.pkl"
with open(cate_id_info_f, 'rb') as f:
    cate_id_info = pickle.load(f)


pano_seg_class = \
  {"Person": [0], "Bench": [13], "bag": [24], "Bottle": [39], "Cup": [41], "chair": [56], "sofa": [57], "bed": [59], "table": [60, 122], "toilet": [61], 
   "television": [62], "Laptop": [63], "Mouse": [64], "Keyboard": [66], "Microwave": [68], "Oven": [69], "fridge": [72], "books": [73], "Cabinet": [121], 
   "Floor": [123, 88, 97, 98, 101, 124], "Wall": [132, 110, 111, 112, 113], "Ceiling": [119], 'Background':[80]}

semantic_id_map = {}
for seg_class in pano_seg_class:
    for sem_id_raw in pano_seg_class[seg_class]:
        semantic_id_map[sem_id_raw] = pano_seg_class[seg_class][0]
def semantic_map(raw_sem_id):
  if raw_sem_id in semantic_id_map:
    return semantic_id_map[raw_sem_id]
  elif raw_sem_id>80:
    return raw_sem_id
  else:
    return raw_sem_id

# map color of some instance to MASK-RCNN CoCo for better visualization 
color_map[60] = [192, 128, 192]  # Dining table
color_map[56] = [192, 64, 64]    # chair
color_map[13] = [64, 128, 128]   # Bench
color_map[62] = [0, 64, 192]     # television
color_map[24] = [128, 64, 192]   # bag
color_map[41] = [64, 128, 64]    # Cup
color_map[72] = [64, 192, 0]     # fridge
color_map[59] = [64, 64, 192]     # bed
color_map[57] = [64, 192, 64]     # couch-sofa

# # definition for evaluation code
# CLASS_LABELS_GT = ['bed', 'chair', 'sofa', 'table', 'fridge', 'television', 'bag']
# CLASS_LABELS_GT.sort()
# VALID_CLASS_IDS_GT = np.array([pano_seg_class[semantic_label][0] for semantic_label in CLASS_LABELS_GT])
# VALID_CLASSID2COLOR = {}
# for id in VALID_CLASS_IDS_GT:
#     VALID_CLASSID2COLOR[id] = color_map[id]

# definition for evaluation in 10 sequences
CLASS_LABELS_GT = ['bed', 'chair', 'sofa', 'table', 'fridge', 'television', 'bag', 'books', 'toilet']
CLASS_LABELS_GT.sort()
VALID_CLASS_IDS_GT = np.array([pano_seg_class[semantic_label][0] for semantic_label in CLASS_LABELS_GT])
VALID_CLASSID2COLOR = {}
for id in VALID_CLASS_IDS_GT:
    VALID_CLASSID2COLOR[id] = color_map[id]
