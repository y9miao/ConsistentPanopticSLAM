scene_folder=$1
orbslam_setting_f=/home/yang/toolbox/test_field/volumetric-semantically-consistent-3D-panoptic-mapping/mapping_ros_ws/src/orbslam_semantic/examples/orbslam_insconv.yaml
img_num=997
# img_num=10
cd /home/yang/toolbox/test_field/volumetric-semantically-consistent-3D-panoptic-mapping/mapping_ros_ws
source ./devel/setup.bash
cd ./src/orbslam_semantic

mkdir ${scene_folder}/orbslam

./examples/rgbd_INSCONV \
./Vocabulary/ORBvoc.txt \
${orbslam_setting_f} \
${scene_folder} \
${img_num} \

# cd /home/yang/toolbox/test_field/volumetric-semantically-consistent-3D-panoptic-mapping/mapping_ros_ws/src/orbslam_semantic/scripts/OrbSlamForINSCONV
# python traj_orbslamToScanNet.py \
# --traj_orbslam ${scene_folder}/orbslam/trajectory_orb_slam.txt \
# --scene_folder ${scene_folder}