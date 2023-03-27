# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...
#
# NOTE: The prediction files must live in the root of the given prediction path.
#       Predicted mask .txt files must live in a subfolder.
#       Additionally, filenames must not contain spaces.
# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *_vh_clean_2.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.
#
# example usage: evaluate_semantic_instance.py --scan_path [path to scan data] --output_file [output file]

# python imports
import math
import os, sys, argparse
import inspect
from copy import deepcopy
import open3d as o3d
import copy
# print(sys.path)

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import SemanticEvaluation.util as util
import SemanticEvaluation.util_3d as util_3d

# ---------- Evaluation params ---------- #

# overlaps for evaluation
overlaps             = np.append(np.arange(0.5,0.95,0.05), 0.25)
# minimum region size for evaluation [verts]
min_region_sizes     = np.array( [ 100 ] )
# distance thresholds [m]
distance_threshes    = np.array( [  float('inf') ] )
# distance confidences
distance_confs       = np.array( [ -float('inf') ] )

# ---------- Label info ---------- #
CLASS_LABELS = None
VALID_CLASS_IDS = None
ID_TO_LABEL = None
LABEL_TO_ID = None
def init(task):
    if(task == "CoCo"):
        from SemanticEvaluation.color_maps import CLASS_LABELS_GT,VALID_CLASS_IDS_GT
    elif(task == "CoCoPano"):
        from SemanticEvaluation.pano_colormap import CLASS_LABELS_GT,VALID_CLASS_IDS_GT
    else:
        raise ValueError(" Not matced task!")
    global CLASS_LABELS
    global VALID_CLASS_IDS
    global ID_TO_LABEL
    global LABEL_TO_ID
    CLASS_LABELS = CLASS_LABELS_GT
    VALID_CLASS_IDS = VALID_CLASS_IDS_GT
    ID_TO_LABEL = {}
    LABEL_TO_ID = {}
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
        ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

def evaluate_matches(matches, use_confidence=False):
    # overlaps = overlaps
    # min_region_sizes = [ min_region_sizes[0] ]
    dist_threshes = [ distance_threshes[0] ]
    dist_confs = [ distance_confs[0] ]
    
    # results: class x overlap, IOU chamfer distance
    chamfer_matches = {}
    panoptic_quality = {}
    gt_class_counts = {}

    ap = np.zeros( (len(dist_threshes) , len(CLASS_LABELS) , len(overlaps)) , np.float )
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            overlap_th_perc = int(overlap_th*100) 
            chamfer_matches[overlap_th_perc] = {}
            panoptic_quality[overlap_th_perc] = {}
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
            for li, label_name in enumerate(CLASS_LABELS):
                panoptic_quality[overlap_th_perc][label_name] = {"PQ":np.nan, "SQ":np.nan, "RQ":np.nan}
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    
                    # filter groups in ground truth
                    gt_instances = [ gt for gt in gt_instances if gt['instance_id']>=1000 and gt['vert_count']>=min_region_size and gt['med_dist']<=distance_thresh and gt['dist_conf']>=distance_conf ]
                    gt_class_counts[label_name] = len(gt_instances)

                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true  = np.ones ( len(gt_instances) )
                    cur_score = np.ones ( len(gt_instances) ) * (-float("inf"))
                    cur_match = np.zeros( len(gt_instances) , dtype=np.bool )
                    cur_IOU = np.zeros( len(gt_instances), dtype=np.float32)
                    # collect matches
                    for (gti,gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (gt['vert_count']+pred['vert_count']-pred['intersection'])
                            if overlap > overlap_th:
                                confidence = None
                                if use_confidence:
                                    confidence = pred['confidence']
                                else:
                                    confidence = 1
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    if( confidence> chamfer_matches[overlap_th_perc][gt['instance_id']]['confidence'] ):
                                        chamfer_matches[overlap_th_perc][gt['instance_id']]['confidence'] = confidence
                                        chamfer_matches[overlap_th_perc][gt['instance_id']]['pred_file'] = pred['filename']

                                    max_score = max( cur_score[gti] , confidence )
                                    min_score = min( cur_score[gti] , confidence )
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true  = np.append(cur_true,0)
                                    cur_score = np.append(cur_score,min_score)
                                    cur_match = np.append(cur_match,True)

                                    cur_IOU[gti] = max( overlap, cur_IOU[gti] )
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                                    cur_IOU[gti] = overlap
                                    # save gt-pred match for chamfer distance calculation
                                    chamfer_matches[overlap_th_perc][gt['instance_id']] = {'gt_file': m, 'pred_file': pred['filename'], 'confidence': confidence}
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true  = cur_true [ cur_match==True ]
                    cur_score = cur_score[ cur_match==True ]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (gt['vert_count']+pred['vert_count']-gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist']>distance_thresh or gt['dist_conf']<distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore)/pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true,0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score,confidence)

                    # append to overall results
                    y_true  = np.append(y_true,cur_true)
                    y_score = np.append(y_score,cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort      = np.argsort(y_score)
                    y_score_sorted      = y_score[score_arg_sort]
                    y_true_sorted       = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds,unique_indices) = np.unique( y_score_sorted , return_index=True )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples      = len(y_score_sorted)
                    num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    precision         = np.zeros(num_prec_recall)
                    recall            = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
                    # deal with remaining
                    for idx_res,idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores-1]
                        tp = num_true_examples - cumsum
                        fp = num_examples      - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p  = float(tp)/(tp+fp)
                        r  = float(tp)/(tp+fn)
                        precision[idx_res] = p
                        recall   [idx_res] = r

                        # calculate panoptic quality, confidence threshold = 0
                        if(idx_res==0):
                            recognition_quality = tp/(tp+0.5*fp+0.5*fn)
                            segment_quality = np.sum(cur_IOU)/tp
                            if tp>0:
                                panoptic_quality[overlap_th_perc][label_name]["SQ"] = segment_quality
                            panoptic_quality[overlap_th_perc][label_name]["RQ"] = recognition_quality
                            panoptic_quality[overlap_th_perc][label_name]["PQ"] = np.sum(cur_IOU)/(tp+0.5*fp+0.5*fn)
                            pass
                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall   [-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv,[-0.5,0,0.5],'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di,li,oi] = ap_current
    return ap, chamfer_matches, gt_class_counts, panoptic_quality

def compute_averages(aps):
    d_inf = 0
    o75   = np.where(np.isclose(overlaps,0.75))
    o50   = np.where(np.isclose(overlaps,0.5))
    o25   = np.where(np.isclose(overlaps,0.25))
    oAllBut25  = np.where(np.logical_not(np.isclose(overlaps,0.25)))
    avg_dict = {}
    #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,o75])
    avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
    avg_dict["classes"]  = {}
    for (li,label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name]             = {}
        #avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,oAllBut25])
        avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,o75])
        avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
        avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
    return avg_dict


def assign_instances_for_scan(pred_file, gt_file, pred_path):
    try:
        pred_info = util_3d.read_instance_prediction_file(pred_file, pred_path)
    except Exception:
        util.print_error('unable to load ' + pred_file )
    try:
        gt_ids = util_3d.load_ids_npy(gt_file).reshape(-1)
    except Exception:
        util.print_error('unable to load ' + gt_file )

    # get gt instances
    gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids//1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    for pred_mask_file in pred_info:
        label_id = int(pred_info[pred_mask_file]['label_id'])
        conf = pred_info[pred_mask_file]['conf']
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = util_3d.load_ids_npy(pred_mask_file).reshape(-1)
        if len(pred_mask) != len(gt_ids):
            util.print_error('wrong number of lines in ' + pred_mask_file + '(%d) vs #mesh vertices (%d), please double check and/or re-download the mesh' % (len(pred_mask), len(gt_ids)))
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < min_region_sizes[0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance['filename'] = pred_mask_file
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection']   = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt

def chamfer_distance(pred_vertice, gt_vertice, dist_th = 1000):
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_vertice)
    kdtree_gt = o3d.geometry.KDTreeFlann(gt_pcd)

    distances_pred2gt = []
    for i, vert in enumerate(pred_vertice):
        _, inds, dist = kdtree_gt.search_knn_vector_3d(vert, 1)
        if dist[0]<dist_th:
            distances_pred2gt.append(np.sqrt(dist[0]))

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_vertice)
    kdtree_pred = o3d.geometry.KDTreeFlann(pred_pcd)

    distances_gt2pred = []
    for i, vert in enumerate(gt_vertice):
        _, inds, dist = kdtree_pred.search_knn_vector_3d(vert, 1)
        if dist[0]<dist_th:
            distances_gt2pred.append(np.sqrt(dist[0]))

    dist_pred2gt = np.average(np.array(distances_pred2gt))
    dist_gt2pred = np.average(np.array(distances_gt2pred))
    return dist_pred2gt, dist_gt2pred

def calculate_chamfer_distance(chamfer_matches, mesh_f):
    chamfer_distances = {}
    
    gt_mesh = o3d.io.read_point_cloud(mesh_f)
    mesh_vertices = np.array(gt_mesh.points)
    gt_labels = None

    chamfer_distances_result = {}
    for ov in [25, 50, 75]:
        chamfer_matches_ov = chamfer_matches[ov]
        
        chamfer_distances_ov = {}
        # if(len(chamfer_matches_ov) == 0):
        #     chamfer_distances[ov] = chamfer_distances_ov
        #     continue
        
        for instance_id in chamfer_matches_ov:
            if gt_labels is None:
                gt_labels_f = chamfer_matches_ov[instance_id]['gt_file']
                gt_labels = np.load(gt_labels_f).reshape(-1)
            
            pred_labels_f = chamfer_matches_ov[instance_id]['pred_file']
            pred_labels = np.load(pred_labels_f).reshape(-1).astype(bool)

            gt_vertice = mesh_vertices[instance_id == gt_labels]
            pred_vertice = mesh_vertices[pred_labels]
            dist_pred2gt, dist_gt2pred = chamfer_distance(pred_vertice, gt_vertice)

            class_name = ID_TO_LABEL[instance_id//1000]
            if class_name not in chamfer_distances_ov:
                chamfer_distances_ov[class_name] = {'pred2gt': [], 'gt2pred': []}
            chamfer_distances_ov[class_name]['pred2gt'].append(dist_pred2gt)
            chamfer_distances_ov[class_name]['gt2pred'].append(dist_gt2pred)

        chamfer_distances[ov] = copy.deepcopy(chamfer_distances_ov)

        chamfer_distances_result[ov]['pred2gt'] = { class_name: np.nan for class_name in CLASS_LABELS}
        chamfer_distances_result[ov]['gt2pred'] = { class_name: np.nan for class_name in CLASS_LABELS}
        for class_name in CLASS_LABELS:
            if class_name in chamfer_distances_ov and len(chamfer_distances_ov[class_name]['pred2gt']!=0):
                chamfer_distances_result[ov]['pred2gt'][class_name] = np.average(np.array(chamfer_distances_ov[class_name]['pred2gt']))
            if class_name in chamfer_distances_ov and len(chamfer_distances_ov[class_name]['gt2pred']!=0):
                chamfer_distances_result[ov]['gt2pred'][class_name] = np.average(np.array(chamfer_distances_ov[class_name]['gt2pred']))
    
    breakpoint = None
    return chamfer_distances_result

def chamfer_distance_cuda(pred_vertice_cuda, gt_vertice_cuda, dist_th = 1000):
    import open3d.core as o3c
    point_dtype = o3c.float32

    knn_pred = o3c.nns.NearestNeighborSearch(pred_vertice_cuda)
    knn_pred.knn_index()
    knn_gt = o3c.nns.NearestNeighborSearch(gt_vertice_cuda)
    knn_gt.knn_index()

    indices, distances_sqr = knn_gt.knn_search(pred_vertice_cuda, 1)
    indices = indices.reshape(-1)
    distances_sqr = distances_sqr.reshape(-1)
    matched = distances_sqr< dist_th**2
    distances_matched_sqr = distances_sqr[matched]
    dist_pred2gt = o3c.Tensor.mean(distances_matched_sqr).cpu().numpy()

    indices, distances_sqr = knn_pred.knn_search(gt_vertice_cuda, 1)
    indices = indices.reshape(-1)
    distances_sqr = distances_sqr.reshape(-1)
    matched = distances_sqr< dist_th**2
    distances_matched_sqr = distances_sqr[matched]
    dist_gt2pred = o3c.Tensor.mean(distances_matched_sqr).cpu().numpy()

    return dist_pred2gt, dist_gt2pred

def calculate_chamfer_distance_cuda(chamfer_matches, mesh_f, device):
    import open3d.core as o3c
    point_dtype = o3c.float32
    label_type = o3c.int64
    bool_type = o3c.bool
    chamfer_distances = {}
    
    gt_mesh = o3d.io.read_point_cloud(mesh_f)
    mesh_vertices = np.array(gt_mesh.points)
    mesh_vertices_cuda = o3c.Tensor(mesh_vertices, dtype=point_dtype, device=device)
    gt_labels = None
    gt_labels_cuda = None

    chamfer_distances_result = {}
    for ov in [25, 50, 75]:
        chamfer_matches_ov = chamfer_matches[ov]
        chamfer_distances_ov = {}

        for instance_id in chamfer_matches_ov:
            if gt_labels is None:
                gt_labels_f = chamfer_matches_ov[instance_id]['gt_file']
                gt_labels = np.load(gt_labels_f).reshape(-1)
                gt_labels_cuda = o3c.Tensor(gt_labels, dtype=label_type, device=device)
            
            pred_labels_f = chamfer_matches_ov[instance_id]['pred_file']
            pred_labels = np.load(pred_labels_f).reshape(-1).astype(bool)
            pred_labels_cuda = o3c.Tensor(pred_labels, dtype=bool_type, device=device)

            gt_vertice_cuda = mesh_vertices_cuda[gt_labels_cuda==instance_id]
            pred_vertice_cuda = mesh_vertices_cuda[pred_labels_cuda]
            dist_pred2gt, dist_gt2pred = chamfer_distance_cuda(pred_vertice_cuda, gt_vertice_cuda)
            
            class_name = ID_TO_LABEL[instance_id//1000]
            if class_name not in chamfer_distances_ov:
                chamfer_distances_ov[class_name] = {'pred2gt': [], 'gt2pred': []}
            chamfer_distances_ov[class_name]['pred2gt'].append(dist_pred2gt)
            chamfer_distances_ov[class_name]['gt2pred'].append(dist_gt2pred)

            if instance_id==58000 and ov==50:
                gt_vertice_cpu = gt_vertice_cuda.cpu().numpy()
                pred_vertice_cpu = pred_vertice_cuda.cpu().numpy()
                np.save("/home/yang/big_ssd/results/206/same_ros2_order_debug/debug/gt_sofa.npy", gt_vertice_cpu)
                np.save("/home/yang/big_ssd/results/206/same_ros2_order_debug/debug/pred_sofa.npy", pred_vertice_cpu)
        
        chamfer_distances[ov] = copy.deepcopy(chamfer_distances_ov)
        chamfer_distances_result[ov] = {}
        chamfer_distances_result[ov]['pred2gt'] = { class_name: np.nan for class_name in CLASS_LABELS}
        chamfer_distances_result[ov]['gt2pred'] = { class_name: np.nan for class_name in CLASS_LABELS}
        for class_name in CLASS_LABELS:
            if class_name in chamfer_distances_ov and len(chamfer_distances_ov[class_name]['pred2gt'])!=0:
                chamfer_distances_result[ov]['pred2gt'][class_name] = np.average(np.array(chamfer_distances_ov[class_name]['pred2gt']))
            if class_name in chamfer_distances_ov and len(chamfer_distances_ov[class_name]['gt2pred'])!=0:
                chamfer_distances_result[ov]['gt2pred'][class_name] = np.average(np.array(chamfer_distances_ov[class_name]['gt2pred']))
    
    breakpoint = None

    return chamfer_distances_result

def print_results(avgs):
    sep     = "" 
    col1    = ":"
    lineLen = 64

    print("")
    print(str("#")*lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    print(line)
    print("#"*lineLen)

    for (li,label_name) in enumerate(CLASS_LABELS):
        ap_avg  = avgs["classes"][label_name]["ap"]
        ap_50o  = avgs["classes"][label_name]["ap50%"]
        ap_25o  = avgs["classes"][label_name]["ap25%"]
        line  = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg ) + sep
        line += sep + "{:>15.3f}".format(ap_50o ) + sep
        line += sep + "{:>15.3f}".format(ap_25o ) + sep
        print(line)

    all_ap_avg  = avgs["all_ap"]
    all_ap_50o  = avgs["all_ap_50%"]
    all_ap_25o  = avgs["all_ap_25%"]

    print(str("-")*lineLen)
    line  = "{:<15}".format("average") + sep + col1 
    line += "{:>15.3f}".format(all_ap_avg)  + sep 
    line += "{:>15.3f}".format(all_ap_50o)  + sep
    line += "{:>15.3f}".format(all_ap_25o)  + sep
    print(line)
    print(" ")


def write_result_file(avgs, chamfer_distance_class, output_folder, evaluation_folder, gt_class_counts, panoptic_quality):
    _SPLITTER = ','
    import pickle
    mAP_txt = os.path.join(output_folder, "mAP.txt")
    mAP_pkl = os.path.join(evaluation_folder, "mAP.pkl")
    chamfer_txt = os.path.join(output_folder, "chamfer.txt")
    chamfer_pkl = os.path.join(evaluation_folder, "chamfer.pkl")
    gt_class_counts_txt = os.path.join(output_folder, "gt_class_counts.txt")
    gt_class_counts_pkl = os.path.join(evaluation_folder, "gt_class_counts.pkl")
    # matched_instance_num_txt = 
    panoptic_quality_txt = os.path.join(output_folder, "panoptic_quality.txt")
    panoptic_quality_pkl = os.path.join(evaluation_folder, "panoptic_quality.pkl")

    with open(gt_class_counts_pkl, 'wb') as f:
        pickle.dump(gt_class_counts, f)
    # test_gt_class_counts = None
    # with open(gt_class_counts_pkl, 'rb') as f:
    #     test_gt_class_counts = pickle.load(f)

    # save mAP results
    with open(mAP_txt, 'w') as f:
        f.write(_SPLITTER.join(['class'.ljust(5), 'class id'.ljust(8), 'ap75'.ljust(10), 
                'ap50'.ljust(10),'ap25'.ljust(10) ]) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap75 = format(avgs["classes"][class_name]["ap"], '.10f').ljust(10)[:10]
            ap50 = format(avgs["classes"][class_name]["ap50%"], '.10f').ljust(10)[:10]
            ap25 = format(avgs["classes"][class_name]["ap25%"], '.10f').ljust(10)[:10]
            f.write(_SPLITTER.join([str(x) for x in [str(class_name).ljust(5), str(class_id).ljust(8), ap75, ap50,ap25]]) + '\n') 


    with open(mAP_pkl, 'wb') as f:
        pickle.dump(avgs, f)
    # test_mAP = None
    # with open(mAP_pkl, 'rb') as f:
    #     test_mAP = pickle.load(f)

    # save chamfer distance results
    with open(chamfer_txt, 'w') as f:
        f.write(_SPLITTER.join(['class'.ljust(12), 'class id'.ljust(8),  
                'pred_ov50'.ljust(8),'gt_ov50'.ljust(8),'pred_ov75'.ljust(8),'gt_ov75'.ljust(8) ]) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            pred_ov50 = format(chamfer_distance_class[50]['pred2gt'][class_name]*1e3, '.10f').ljust(8)[:8]
            pred_ov75 = format(chamfer_distance_class[75]['pred2gt'][class_name]*1e3, '.10f').ljust(8)[:8]
            gt_ov50 = format(chamfer_distance_class[50]['gt2pred'][class_name]*1e3, '.10f').ljust(8)[:8]
            gt_ov75 = format(chamfer_distance_class[75]['gt2pred'][class_name]*1e3, '.10f').ljust(8)[:8]

            f.write(_SPLITTER.join([str(x) for x in [str(class_name).ljust(12), str(class_id).ljust(8), 
                pred_ov50,gt_ov50 ,pred_ov75, gt_ov75]]) + '\n')    
            
    with open(chamfer_pkl, 'wb') as f:
        pickle.dump(chamfer_distance_class, f)

    # savepanoptic quality results
    with open(panoptic_quality_txt, 'w') as f:
        f.write(_SPLITTER.join(['class'.ljust(12), 'class id'.ljust(8),  
                'PQ_50'.ljust(5),'SQ_50'.ljust(5),'RQ_50'.ljust(5),
                'PQ_75'.ljust(5),'SQ_75'.ljust(5),'RQ_75'.ljust(5) ]) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            PQ_50 = format(panoptic_quality[50][class_name]["PQ"], '.5f').ljust(5)[:5]
            SQ_50 = format(panoptic_quality[50][class_name]["SQ"], '.5f').ljust(5)[:5]
            RQ_50 = format(panoptic_quality[50][class_name]["RQ"], '.5f').ljust(5)[:5]
            PQ_75 = format(panoptic_quality[75][class_name]["PQ"], '.5f').ljust(5)[:5]
            SQ_75 = format(panoptic_quality[75][class_name]["SQ"], '.5f').ljust(5)[:5]
            RQ_75 = format(panoptic_quality[75][class_name]["RQ"], '.5f').ljust(5)[:5]

            f.write(_SPLITTER.join([str(x) for x in [str(class_name).ljust(12), str(class_id).ljust(8), 
                PQ_50, SQ_50, RQ_50,PQ_75, SQ_75, RQ_75]]) + '\n')    
    
    with open(panoptic_quality_pkl, 'wb') as f:
        pickle.dump(panoptic_quality, f)
    test_panoptic_quality = None
    with open(panoptic_quality_pkl, 'rb') as f:
        test_panoptic_quality = pickle.load(f)
        print("panoptic_quality_pkl: ", panoptic_quality_pkl)
        print("test_panoptic_quality", test_panoptic_quality[50])
    return None


def evaluate(pred_files, gt_files, pred_path, output_folder, evaluation_folder, gt_mesh_f):
    print('evaluating', len(pred_files), 'scans...')
    matches = {}
    for i in range(len(pred_files)):
        matches_key = os.path.abspath(gt_files[i])
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(pred_files[i], gt_files[i], pred_path)
        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt
        sys.stdout.write("\rscans processed: {}".format(i+1))
        sys.stdout.flush()
    print('')
    ap_scores, chamfer_matches,gt_class_counts, panoptic_quality = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    import open3d.core as o3c
    chamfer_distances = None
    if(o3c.cuda.is_available()):
        device = o3c.Device("CUDA", 0)
        chamfer_distances = calculate_chamfer_distance_cuda(chamfer_matches, gt_mesh_f, device)
    else:
        chamfer_distances = calculate_chamfer_distance(chamfer_matches, gt_mesh_f)
    # print
    print_results(avgs)
    write_result_file(avgs, chamfer_distances, output_folder,evaluation_folder, gt_class_counts, panoptic_quality)

