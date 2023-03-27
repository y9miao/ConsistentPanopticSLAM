import numpy as np
import os
from plyfile import PlyData, PlyElement
import trimesh
from scipy import stats
import EvaluationMy.GeometricEvaluation.geo_utils as GEO

def GetObjIDFromXml(config_gt):
    ObjID = {}
    for class_label in config_gt["CLASS_LABELS_GT"]:
        ObjID[class_label] = []

    import xml.etree.ElementTree as ET
    tree = ET.parse(config_gt["xml_f"])
    root = tree.getroot()
    # if objects are what we want for evaluation, record the label_id
    for obj_label in root:
        obj_class = obj_label.attrib['nyu_class']
        obj_class_annotation = obj_label.attrib['text']
        if obj_class:
            if obj_class_annotation:
                if(obj_class in ObjID):
                    obg_id = obj_label.attrib['id']
                    ObjID[obj_class].append(np.uint32(obg_id))
    return ObjID


def GroundTruthSceneNN2ScanNet(config_gt):
    # eval_folder = config_gt["out_folder"]
    # if not os.path.isdir(eval_folder):
    #     os.mkdir(eval_folder)
    # gt_folder = os.path.join(config_gt["out_folder"],"gt")
    # if not os.path.isdir(gt_folder):
    #     os.mkdir(gt_folder)
    gt_folder = config_gt["gt_folder"]
    if os.path.isdir(gt_folder) and len(list(os.listdir(gt_folder)))!=0:
        return None
    else:
        os.mkdir(gt_folder)

    ObjID = GetObjIDFromXml(config_gt)
    ObjID2InstanceID = {}

    for class_label in ObjID:
        class_id = config_gt["LABELS2IDsMap_GT"][class_label]
        inclass_obj_num = 0

        for obj in ObjID[class_label]:
            InstanceID = int(class_id*1000+inclass_obj_num)
            ObjID2InstanceID[obj] = InstanceID
            inclass_obj_num += 1

    # read in ply vertexes
    plydata = PlyData.read(config_gt["ply_f"])
    vertex_num = plydata['vertex'].count
    vertex_labels = plydata['vertex'].data['label']
    # assign instance id to vertexes
    out_array = np.zeros((vertex_num,1)).astype(int)
    for obj in ObjID2InstanceID:
        out_array[vertex_labels==obj] = ObjID2InstanceID[obj]

    scene_f = "sceneNN"+str(config_gt["scene_num"])+".npy"
    out_gt_f = os.path.join(gt_folder, scene_f)
    np.save(out_gt_f, out_array)

    return ObjID2InstanceID


def Pred2ScanNetO3d(config_pred):
    # make directories
    eval_folder = config_pred["out_folder"]
    if not os.path.isdir(eval_folder):
        os.mkdir(eval_folder)
    pred_folder = os.path.join(config_pred["out_folder"],"pred")
    if not os.path.isdir(pred_folder):
        os.mkdir(pred_folder)
    pred_mask_folder = os.path.join(pred_folder,"predicted_masks")
    if not os.path.isdir(pred_mask_folder):
        os.mkdir(pred_mask_folder)

    # extract semantic information from colors of semantic ply result
    import open3d as o3d
    semantic_mesh = o3d.io.read_point_cloud(config_pred["semantic_ply_f"])
    
    vertex_num = len(semantic_mesh.points)
    colors = np.array(semantic_mesh.colors)
    colors_uint = (colors*255.0).astype(np.uint8)

    vertex_sematic_label = np.zeros(vertex_num)
    for class_id in config_pred["VALID_CLASS_IDS_GT"]:
        class_color = config_pred["VALID_CLASSID2COLOR"][class_id]
        class_rgb_align = (colors_uint == class_color).all(axis=1)
        vertex_sematic_label[class_rgb_align] = class_id
    # semantics = np.unique(vertex_sematic_label)

    # associate semantic and instance information
    mask_files = []
    instance_mesh = o3d.io.read_point_cloud(config_pred["instance_ply_f"])
    instance_colors = np.array(instance_mesh.colors)
    instance_colors_uint = (instance_colors*255.0).astype(np.uint8)
    instance_colors_unique = np.unique(instance_colors_uint, axis=0)
    instances = {}
    instance_id = 1
    semantic_max_vertices = {}
    for color_i in range(instance_colors_unique.shape[0]):
        color = instance_colors_unique[color_i]
        if np.array_equal(color[:3], [200,200,200]) or np.array_equal(color[:3], [0,0,0]):
            continue
        instance_mask = (instance_colors_uint==color).all(axis=1)
        instance_class = stats.mode(vertex_sematic_label[instance_mask])[0][0].astype(int)
        if(instance_class!=0):
            instances[instance_id] = {'mask':instance_mask, 'class':instance_class,'confidence':0, 'file': None}
            instance_id += 1
        if instance_class not in semantic_max_vertices:
            semantic_max_vertices[instance_class] = np.sum(instance_mask)
        else:
            semantic_max_vertices[instance_class] = max(semantic_max_vertices[instance_class], np.sum(instance_mask))
    # calculate confidence for each instance
    for instance_id in instances.keys():
        instances[instance_id]['confidence'] = np.sum(instances[instance_id]['mask'])*1.0/semantic_max_vertices[instances[instance_id]['class']]

    scene_f = "sceneNN"+str(config_pred["scene_num"])+".txt"
    scene_f = os.path.join(pred_folder,scene_f)
    instance_files = []

    # write scene files
    with open(scene_f, 'w') as f:
        for instance_id in instances.keys():
            instance_f = "sceneNN"+str(config_pred["scene_num"])+"_%d_%d.npy"%(instances[instance_id]['class'],instance_id)
            instance_files.append(instance_f)

            instances[instance_id]['file'] = os.path.join(pred_mask_folder,instance_f)

            f.write("predicted_masks/"+instance_f)
            f.write(" "+str(instances[instance_id]['class']))
            f.write(" %f\n"%(instances[instance_id]['confidence']))
        f.close()

    for instance_id in instances.keys():
        if(instances[instance_id]['file'] is not None):
            mask_to_write = instances[instance_id]['mask'].reshape(-1,1)
            np.save(instances[instance_id]['file'],mask_to_write)

    return instances