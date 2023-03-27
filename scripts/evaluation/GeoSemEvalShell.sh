#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:${COMMAND_WS}/evaluation/EvaluationMy

SceneNum=$1
ResultFoldder=$2
GTFolder=$3
UseSemantic=$4
Task=$5
EvaluationFolder=geo_sem_eval
# EvaluationFolder=geo_sem_eval_test


echo "UseSemantic: $UseSemantic"

python ${EVAL_WS}/EvaluationMy/GeoSemEval.py \
	--scene_num ${SceneNum} \
	--result_folder ${ResultFoldder} \
	--evaluation_folder ${ResultFoldder}/${EvaluationFolder} \
	--evaluation_metric_folder ${ResultFoldder} \
	--gt_mesh_f ${GTFolder}/${SceneNum}.ply \
	--gt_annotation_f ${GTFolder}/${SceneNum}.xml \
	--gt_mask_folder ${GTFolder}/gt \
	--instance_mesh ${ResultFoldder}/instance_mesh_.ply \
	--semantic_mesh ${ResultFoldder}/semantic_mesh_.ply \
	--use_semantic $UseSemantic \
	--task ${Task} \
