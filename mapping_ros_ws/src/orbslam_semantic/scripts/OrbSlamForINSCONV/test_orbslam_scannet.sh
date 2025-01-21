scene_folder=/home/yang/990Pro/scannet_seqs/data/val/scene0011_00
orbslam_setting_f=/home/yang/990Pro/scannet_seqs/data/val/scene0011_00/orbslam_archive/orbslam.yaml
img_num=2000

cd /home/yang/toolbox/test_field/volumetric-semantically-consistent-3D-panoptic-mapping/mapping_ros_ws
source ./devel/setup.bash
cd ./src/orbslam_semantic

mkdir ${scene_folder}/orbslam

# ./examples/rgbd_scannet \
# ./Vocabulary/ORBvoc.txt \
# ${orbslam_setting_f} \
# ${scene_folder} \
# ${img_num} \

# cd /home/yang/toolbox/test_field/volumetric-semantically-consistent-3D-panoptic-mapping/mapping_ros_ws/src/orbslam_semantic/scripts/OrbSlamForScanNet
# python traj_orbslamToScanNet.py \
# --traj_orbslam ${scene_folder}/orbslam/trajectory_orb_slam.txt \
# --traj_gt ${scene_folder}/trajectory.log \
# --scene_folder ${scene_folder}