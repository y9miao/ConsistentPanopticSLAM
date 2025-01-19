CLASS_LABELS = None
VALID_CLASS_IDS = None
ID_TO_LABEL = None
LABEL_TO_ID = None
COLOR_MAP = None
BackGroundID = None
IsThing = None


def init(task):
    if(task == "CoCoPano"):
        from evaluation_scannet.SemanticEvaluation.scannet_coco_color_map \
            import CLASS_LABELS_GT,VALID_CLASS_IDS_GT, BackgroundSemId, \
                    color_map, IsSemThing
    if(task == "Nyu40"):
        from evaluation_scannet.SemanticEvaluation.scannet_nyu40_colormap \
            import CLASS_LABELS_GT,VALID_CLASS_IDS_GT, BackgroundSemId, \
                    color_map, IsSemThing
    else:
        raise ValueError(" Not matced task!")
    global CLASS_LABELS
    global VALID_CLASS_IDS
    global ID_TO_LABEL
    global LABEL_TO_ID
    global BackGroundID
    global IsThing
    global COLOR_MAP
    CLASS_LABELS = CLASS_LABELS_GT
    VALID_CLASS_IDS = VALID_CLASS_IDS_GT
    ID_TO_LABEL = {}
    LABEL_TO_ID = {}
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
        ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
    COLOR_MAP = color_map
    BackGroundID = BackgroundSemId
    IsThing = IsSemThing