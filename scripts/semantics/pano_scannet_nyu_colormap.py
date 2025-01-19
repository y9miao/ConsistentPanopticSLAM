import numpy as np

BackgroundSemId = 0

color_map = [
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
    1: {'name':"wall", 'isthing':False, 'color': color_map[1]},
    2: {'name':"floor", 'isthing':False, 'color': color_map[2]},
    3: {'name':"cabinet", 'isthing':True, 'color': color_map[3]},
    4: {'name':"bed", 'isthing':True, 'color': color_map[4]},
    5: {'name':"chair", 'isthing':True, 'color': color_map[5]},
    6: {'name':"sofa", 'isthing':True, 'color': color_map[6]},
    7: {'name':"table", 'isthing':True, 'color': color_map[7]},
    8: {'name':"door", 'isthing':True, 'color': color_map[8]},
    9: {'name':"window",'isthing':True, 'color': color_map[9]},
    10: {'name':"bookshelf",'isthing':True, 'color': color_map[10]},
    11: {'name':"picture",'isthing':True, 'color': color_map[11]},
    12: {'name':"counter",'isthing':True, 'color': color_map[12]},
    13: {'name':"blinds",'isthing':False, 'color': color_map[13]},
    14: {'name':"desk",'isthing':True, 'color': color_map[14]},
    15: {'name':"shelves",'isthing':False, 'color': color_map[15]},
    16: {'name':"curtain",'isthing':True, 'color': color_map[16]},
    17: {'name':"dresser",'isthing':False, 'color': color_map[17]},
    18: {'name':"pillow",'isthing':True, 'color': color_map[18]},
    19: {'name':"mirror",'isthing':False, 'color': color_map[19]},
    20: {'name':"floor mat",'isthing':False, 'color': color_map[20]},
    21: {'name':"clothes",'isthing':False, 'color': color_map[21]},
    22: {'name':"ceiling",'isthing':False, 'color': color_map[22]},
    23: {'name':"books",'isthing':False, 'color': color_map[23]},
    24: {'name':"refridgerator",'isthing':True, 'color': color_map[24]},
    25: {'name':"television",'isthing':True, 'color': color_map[25]},
    26: {'name':"paper",'isthing':False, 'color': color_map[26]},
    27: {'name':"towel",'isthing':False, 'color': color_map[27]},
    28: {'name':"shower curtain",'isthing':True, 'color': color_map[28]},
    29: {'name':"box",'isthing':False, 'color': color_map[29]},
    30: {'name':"whiteboard",'isthing':False, 'color': color_map[30]},
    31: {'name':"person",'isthing':True, 'color': color_map[31]},
    32: {'name':"nightstand",'isthing':True, 'color': color_map[32]},
    33: {'name':"toilet",'isthing':True, 'color': color_map[33]},
    34: {'name':"sink",'isthing':True, 'color': color_map[34]},
    35: {'name':"lamp",'isthing':True, 'color': color_map[35]},
    36: {'name':"bathtub",'isthing':True, 'color': color_map[36]},
    37: {'name':"bag",'isthing':True, 'color': color_map[37]},
    38: {'name':"otherstructure",'isthing':False, 'color': color_map[38]},
    39: {'name':"otherfurniture",'isthing':True, 'color': color_map[39]},
    40: {'name':"otherprop",'isthing':False, 'color': color_map[40]},
}

def semantic_map(raw_sem_id):
    if raw_sem_id != BackgroundSemId:
        return raw_sem_id+1
    else:
        return raw_sem_id

CLASS_LABELS_GT = [SCANNET_SEM_SEG_CATEGORIES[cate_id]['name'] for cate_id in SCANNET_SEM_SEG_CATEGORIES]
CLASS_LABELS_GT.sort()
VALID_CLASS_IDS_GT = np.array(list(SCANNET_SEM_SEG_CATEGORIES.keys()))
VALID_CLASSID2COLOR = {}
for cate_id in VALID_CLASS_IDS_GT:
    VALID_CLASSID2COLOR[id] = SCANNET_SEM_SEG_CATEGORIES[cate_id]['color']

VALID_CLASSLABEL2ID = {}
for cate_id in VALID_CLASS_IDS_GT:
    class_label = SCANNET_SEM_SEG_CATEGORIES[cate_id]['name']
    VALID_CLASSLABEL2ID[class_label] = cate_id

VALID_ID2CLASSLABEL = {}
for cate_id in VALID_CLASS_IDS_GT:
    class_label = SCANNET_SEM_SEG_CATEGORIES[cate_id]['name']
    VALID_ID2CLASSLABEL[cate_id] = class_label
