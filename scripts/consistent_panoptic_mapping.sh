#!/bin/bash
SceneNum=$1
DataFolder=$2
ResultFoldder=$3
IntermediateSegsFolders=$4
ThreadNum=$5
logTest=$6
step=$7

export PYTHONPATH=${PYTHONPATH}:mapping_ros_ws/devel/lib

python scripts/panoptic_mapping_.py \
--scene_num ${SceneNum} \
--data_folder ${DataFolder} \
--result_folder ${ResultFoldder} \
--start 1 \
--end -1 \
--data_association 2 \
--inst_association 4 \
--seg_graph_confidence 3 \
--step ${step} \
--task cocoPano \
--temporal_results \
--intermediate_seg_folder ${IntermediateSegsFolders} \
--num_threads ${ThreadNum} \
--log "${logTest}"
