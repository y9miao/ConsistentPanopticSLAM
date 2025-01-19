import detectron2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

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

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    meta["categories"] = SCANNET_SEM_SEG_CATEGORIES

    return meta

def register_scannet_panoptic_metadata():
    meta = getMetaData()
    # register for metadata
    thing_class_names = []
    for set_split in ['train', 'val']:
        data_set_name = "scannet_"+set_split
        MetadataCatalog.get(data_set_name).set(
            thing_classes = meta['thing_classes'],
            thing_colors = meta['thing_colors'],
            stuff_classes = meta['stuff_classes'],
            stuff_colors = meta['stuff_colors'],
            ignore_label = 255,
            evaluator_type="scannet_panoptic_seg",
            thing_dataset_id_to_contiguous_id = meta['thing_dataset_id_to_contiguous_id'],
            stuff_dataset_id_to_contiguous_id = meta['stuff_dataset_id_to_contiguous_id'],
            categories = meta["categories"], 
        )

def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    from detectron2.config import CfgNode as CN
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


class Mask2FormerWrapper(object):

    def __init__(self, cfg_file, task="Pano_seg", score_thresh=0.5):
        self.task_ = task
        self.cpu_device = torch.device("cpu")
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        try:
            cfg.merge_from_file(cfg_file)
            cfg.freeze()
        except:
            raise Exception("Do not support type `{}`".format(task))
        
        # register meta data for scannet categories
        register_scannet_panoptic_metadata()
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.predictor_ = DefaultPredictor(cfg)
    

    def predict(self, img):
        predictions = self.predictor_(img)
        # calculate areas for each instance
        if len(predictions['panoptic_seg'][1]) > 0:
            IsAreaInfo = True
            for inst_info in predictions['panoptic_seg'][1]:
                if 'area' not in inst_info:
                    IsAreaInfo = False
                    break
            if not IsAreaInfo:
                ids, counts = torch.unique(predictions['panoptic_seg'][0], return_counts = True)
                ids_arr = ids.cpu().detach().numpy()
                counts_arr = counts.cpu().detach().numpy()
                id_count_dict = { ids_arr[idx]: counts_arr[idx] for idx in range(ids_arr.shape[0]) }
                for inst_info in predictions['panoptic_seg'][1]:
                    inst_info['area'] = int(id_count_dict[inst_info['id']])
        return predictions


    