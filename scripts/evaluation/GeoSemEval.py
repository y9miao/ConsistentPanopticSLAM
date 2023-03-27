import os
import sys
import yaml
import argparse
import numpy as np

import GeometricEvaluation.geo_utils as GEO
import SemanticEvaluation.sem_utils as SEM


def parse_args():
    data_path = os.environ.get('DATA')
    parse = argparse.ArgumentParser(description='Evaluation-Python') 
    # files path
    parse.add_argument("--scene_num", type=str, required=True, help="which scene for mapping ")
    parse.add_argument("--result_folder", type=str,required=True,help="folder of mapping results")
    parse.add_argument("--evaluation_folder", type=str,required=True,help="folder of intermediate file for metric")
    parse.add_argument("--evaluation_metric_folder", type=str,required=True,help="folder of intermediate file for metric")

    parse.add_argument("--gt_mesh_f", type=str,required=True,help="gt mesh file")
    parse.add_argument("--gt_annotation_f", type=str,required=True,help="gt annotation file")
    parse.add_argument("--gt_mask_folder", type=str,required=True,help="gt mask folder")

    parse.add_argument("--instance_mesh", type=str, default="", help="result mesh ")
    parse.add_argument("--semantic_mesh", type=str, default="", help="result mesh ")
    parse.add_argument("--label_mesh", type=str, default="", help="result mesh ")
    parse.add_argument("--instance_mesh_proj_name", type=str, default="instance_mesh_proj.ply", help="projected mesh ")
    parse.add_argument("--semantic_mesh_proj_name", type=str, default="semantic_mesh_proj.ply", help="projected mesh ")
    parse.add_argument("--label_mesh_proj_name", type=str, default="label_mesh_proj.ply", help="projected mesh ")
    parse.add_argument("--project_label_mesh", action='store_true', help="whether to project label mesh ")

    parse.add_argument("--use_semantic", type=int, default=0, help="whether to generate files for semantic evaluation ")
    parse.add_argument('--task', type=str, default="CoCo", help="semantic segmentation task")

    return parse.parse_args()

def main():
    args = parse_args()
    # print("args: ", args)

    task = args.task
    if(task == "CoCo"):
        from SemanticEvaluation.color_maps import CLASS_LABELS_GT,VALID_CLASS_IDS_GT,VALID_CLASSID2COLOR
    elif(task == "CoCoPano"):
        from SemanticEvaluation.pano_colormap import CLASS_LABELS_GT,VALID_CLASS_IDS_GT,VALID_CLASSID2COLOR
    else:
        raise ValueError(" Not matced task!")
   
    # resolve the paths
    gt_mesh_f = args.gt_mesh_f
    gt_annotation_f = args.gt_annotation_f
    gt_folder = args.gt_mask_folder
    result_folder = args.result_folder
    result_instance_f = args.instance_mesh
    result_semantic_f = args.semantic_mesh
    result_label_f = args.label_mesh

    evaluation_folder = args.evaluation_folder
    evaluation_metric_folder = args.evaluation_metric_folder
    if not os.path.exists(evaluation_folder):
        os.makedirs(evaluation_folder)
    if not os.path.exists(evaluation_metric_folder):
        os.makedirs(evaluation_metric_folder)

    geo_metrics_f = os.path.join(evaluation_metric_folder,"geometric_evaluation.yaml")

    scene_num = args.scene_num
    use_semantic = args.use_semantic
    
    # implement geometric evaluation
    # print("Geo_metrics calculation")
    # geo_metrics = [GEO.eval_mesh(result_instance_f, gt_mesh_f)]
    # print("     geo_metrics: ", geo_metrics)
    # with open(geo_metrics_f, 'w') as outfile:
    #     yaml.dump(geo_metrics, outfile, default_flow_style=False)

    print("use_semantic: ", use_semantic)
    if(use_semantic==0):
        return None
    # project result meshes to gt meshes for further semantic evaluation 
    print("     Transforming result meshes to the gt vertices")
    instance_proj_name = args.instance_mesh_proj_name
    semantic_proj_name = args.semantic_mesh_proj_name
    label_proj_name = args.label_mesh_proj_name
    result_instance_proj_f = os.path.join(evaluation_folder,instance_proj_name)
    result_semantic_proj_f = os.path.join(evaluation_folder,semantic_proj_name)
    result_label_proj_f = os.path.join(evaluation_folder,label_proj_name)

    import open3d.core as o3c
    if(o3c.cuda.is_available()):
        device = o3c.Device("CUDA", 0)
        GEO.project_to_mesh_cuda(result_instance_f, gt_mesh_f, result_instance_proj_f, device)
        GEO.project_to_mesh_cuda(result_semantic_f, gt_mesh_f, result_semantic_proj_f, device)
        if(args.project_label_mesh):
            GEO.project_to_mesh_cuda(result_label_f, gt_mesh_f, result_label_proj_f, device)
    else:
        GEO.project_to_mesh(result_instance_f, gt_mesh_f, result_instance_proj_f)
        GEO.project_to_mesh(result_semantic_f, gt_mesh_f, result_semantic_proj_f) 
        if(args.project_label_mesh):
            GEO.project_to_mesh(result_label_f, gt_mesh_f, result_label_proj_f)

    LABELS2IDsMap_GT = {}
    for ind,label in enumerate(CLASS_LABELS_GT):
        LABELS2IDsMap_GT[label] = VALID_CLASS_IDS_GT[ind]

    config_gt = {
        "ply_f": gt_mesh_f,
        "xml_f": gt_annotation_f,
        "gt_folder": gt_folder,
        "CLASS_LABELS_GT": CLASS_LABELS_GT,
        "VALID_CLASS_IDS_GT": VALID_CLASS_IDS_GT,
        "LABELS2IDsMap_GT":LABELS2IDsMap_GT,
        "scene_num": scene_num
    }
    config_pred = {
        "instance_ply_f": result_instance_proj_f,
        "semantic_ply_f": result_semantic_proj_f,
        "out_folder": evaluation_folder,
        "CLASS_LABELS_GT": CLASS_LABELS_GT,
        "VALID_CLASS_IDS_GT": VALID_CLASS_IDS_GT,
        "VALID_CLASSID2COLOR": VALID_CLASSID2COLOR,
        "LABELS2IDsMap_GT": LABELS2IDsMap_GT,
        "scene_num": scene_num
    }
    # transform meshes to txt used for semantic evaluation
    print("     writing results to txt for Scannet evaluation")

    ObjID2InstanceID = SEM.GroundTruthSceneNN2ScanNet(config_gt)
    instances = SEM.Pred2ScanNetO3d(config_pred)

    from EvaluationMy.SemanticEvaluation import evaluate_semantic_instance_my
    evaluate_semantic_instance_my.init(task)
    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)
    import SemanticEvaluation.util as util
    import SemanticEvaluation.util_3d as util_3d
    
    # gt_path = os.path.join(config_gt["out_folder"],"gt")
    gt_path = config_gt["gt_folder"]
    
    pred_path = os.path.join(config_pred["out_folder"],"pred")
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('.txt') ]
    gt_files = []
    for i in range(len(pred_files)):
        gt_file = os.path.join(gt_path, pred_files[i].split('.txt')[0]+'.npy')
        if not os.path.isfile(gt_file):
            util.print_error('Result file {} does not match any gt file'.format(pred_files[i]), user_fault=True)
        gt_files.append(gt_file)
        pred_files[i] = os.path.join(pred_path, pred_files[i])

    evaluation_output_folder = result_folder
    evaluation_folder = evaluation_folder
    evaluate_semantic_instance_my.evaluate(pred_files, gt_files, pred_path, evaluation_output_folder,evaluation_folder, gt_mesh_f)


    return None


if __name__=="__main__":
    main()
