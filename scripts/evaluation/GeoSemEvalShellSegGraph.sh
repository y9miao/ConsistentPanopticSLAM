#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:${COMMAND_WS}/evaluation/EvaluationMy

SceneNum=$1
ResultFoldder=$2
GTFolder=$3
UseSemantic=$4
Task=$5
BaseFolder=$6
EvaluationFolder=geo_sem_eval

echo "UseSemantic: $UseSemantic"

python ${EVAL_WS}/EvaluationMy/GeoSemEvalSegGraph.py \
	--scene_num ${SceneNum} \
	--result_folder ${ResultFoldder} \
	--evaluation_folder ${ResultFoldder}/${EvaluationFolder} \
	--evaluation_metric_folder ${ResultFoldder} \
	--gt_mesh_f ${GTFolder}/${SceneNum}.ply \
	--gt_annotation_f ${GTFolder}/${SceneNum}.xml \
	--gt_mask_folder ${GTFolder}/gt_pano \
	--instance_mesh_proj ${ResultFoldder}/${EvaluationFolder}/instance_mesh_proj.ply \
	--base_folder ${BaseFolder} \
	--semantic_mesh_proj ${BaseFolder}/${EvaluationFolder}/semantic_mesh_proj.ply \
	--use_semantic $UseSemantic \
	--task ${Task}