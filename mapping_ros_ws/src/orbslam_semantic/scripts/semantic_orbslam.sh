scene_folder=$1
segments_folder=$2
orbslam_setting_f=$3
semantic_setting_f=$4
img_num=$5
result_folder=$6

cd /home/yang/toolbox/test_field/volumetric-semantically-consistent-3D-panoptic-mapping/mapping_ros_ws
source ./devel/setup.bash

cd ./src/orbslam_semantic

./examples/semantic_rgbd_scannet \
./Vocabulary/ORBvoc.txt \
${segments_folder} \
${orbslam_setting_f} \
${semantic_setting_f} \
${scene_folder} \
${img_num} \
${result_folder}

mkdir ${result_folder}/cfg
cp $orbslam_setting_f ${result_folder}/cfg
cp $semantic_setting_f ${result_folder}/cfg

# ldd ./examples/semantic_rgbd_scannet
