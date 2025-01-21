/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<fstream>
#include<algorithm>
#include<experimental/filesystem>
#include<chrono>

#include<opencv2/core/core.hpp>
#include <opencv2/rgbd.hpp>
#include <Eigen/Core>

#include<System.h>
#include<SystemSemantic.h>
#include<consistent_mapping/segment_confidence.h>

#include <highfive/H5Easy.hpp>

#define COMPILEDWITHC11

using namespace std;

void LoadImagesPath(const string &sceneFolder, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, int img_num);
cv::Mat LoadCamera(const string &sceneFolder);

string zfillString(const string &string_to_fill, std::size_t fill_num);
void LoadSegmentsPath(const string &segmentFolder, vector<string> &vstrSegInfoFiles,
                vector<string> &vstrSegMaskFiles, int img_num);
bool LoadSegments(const string &mask_file, const string &segs_info_file, 
    const string &depth_file, const cv::Mat &K, cv::Mat &segments_mask, 
    map<uint8_t, uint8_t>& segs_semantic_map, 
    std::vector<voxblox::SegmentConfidence*> & segments);
void TestLoadSegments(vector<string> &vstrSegInfoFiles,vector<string> &vstrSegMaskFiles,
    vector<string> &vstrImageFilenamesD, const cv::Mat &K, int img_idx);

int main(int argc, char **argv)
{
    if(argc != 8)
    {
        cerr << endl << "Usage: ./semantic_rgbd_scannet path_to_vocabulary " \
        "path_to_segments orbslam_settings_file semantic_mapping_settings_file " \
        "scene_folder img_num result_folder" << endl;
        return 1;
    }

    std::string vocab_f = std::string(argv[1]);
    std::string segments_folder = std::string(argv[2]);
    std::string orbslam_settings_f = std::string(argv[3]);
    std::string semantic_mapping_settings_f = std::string(argv[4]);
    std::string scene_folder = string(argv[5]);
    int img_num = std::stoi(argv[6]);
    std::string result_folder = string(argv[7]);

    std::string traj_folder = result_folder + "/traj";
    std::string command = "mkdir -p " + traj_folder;
    system(command.c_str());
    string traject_f = traj_folder + "/trajectory_orb_slam.txt";

    // load images
    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    LoadImagesPath(scene_folder, vstrImageFilenamesRGB, vstrImageFilenamesD, img_num);

    // load Segments
    // load camera intrinsics
    cv::Mat K = LoadCamera(scene_folder);
    // retrieve pathes to segments files
    vector<string> vstrSegInfoFiles;
    vector<string> vstrSegMaskFiles;
    LoadSegmentsPath(segments_folder, vstrSegInfoFiles,vstrSegMaskFiles, img_num);
    // TestLoadSegments(vstrSegInfoFiles, vstrSegMaskFiles, vstrImageFilenamesD, K, 1);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::SystemSemantic SemanticSLAM(vocab_f,orbslam_settings_f, 
        semantic_mapping_settings_f, ORB_SLAM3::System::RGBD, false);
    float imageScale = SemanticSLAM.GetImageScale();
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    double fps = 15.0;
    int cout_step = 100; 
    cv::Mat imRGB, imD, segment_mask;
    std::vector<voxblox::SegmentConfidence*> segments_ptrs;
    std::map<uint8_t, uint8_t> segs_semantics_map;

    // rough measure of run rime 
    std::chrono::duration<double> total_run_time(0);
    std::chrono::duration<double> load_time(0);
    std::chrono::duration<double> localization_time(0);
    std::chrono::duration<double> semantic_time(0);
    for(int ni=0; ni<nImages; ni++)
    {
        if(ni%cout_step == 0)
        {
            std::cout << "Processing frame id " << ni <<  std::endl;
        }

        auto frame_start = std::chrono::system_clock::now();

        // Read image and depthmap from file
        imRGB = cv::imread(vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imD = cv::imread(vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double frame_timestamp = double(ni) / fps;
        if(imRGB.empty() || imD.empty())
        {
            cerr << endl << "Failed to load image at frame " << ni << endl;
            return 1;
        }
        if(imageScale != 1.f) // resize image if needed
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        // load segments        
        bool load_segs_success = LoadSegments(
            vstrSegMaskFiles[ni], vstrSegInfoFiles[ni],
            vstrImageFilenamesD[ni], K, segment_mask, 
            segs_semantics_map, segments_ptrs);

        auto load_end = std::chrono::system_clock::now();
        load_time += std::chrono::duration<double>(load_end - frame_start);

        // Track the frame 
        bool use_semantics = false;
        if(load_segs_success)
        {
            auto track_start = std::chrono::system_clock::now();
            Sophus::SE3f SE3_G_C_orbslam = (SemanticSLAM.TrackRGBDSemantic(imRGB,imD,segment_mask, 
                segs_semantics_map, frame_timestamp)).inverse();
            auto track_end = std::chrono::system_clock::now();
            localization_time += std::chrono::duration<double>(track_end-track_start);

            // if track success, then integrate segments
            auto semantic_mapping_start = std::chrono::system_clock::now();
            ORB_SLAM3::Tracking::eTrackingState tracking_state = SemanticSLAM.getTrackingState();
            if(tracking_state == ORB_SLAM3::Tracking::OK)
            {
                voxblox::Transformation T_G_C_orbslam(SE3_G_C_orbslam.matrix());
                // set orbslam estimated poses as segments's poses
                for(SegmentConfidence * seg_ptr : segments_ptrs)
                {
                    seg_ptr->T_G_C_ = T_G_C_orbslam;
                }
                // integrate segments into semantic maps
                SemanticSLAM.IntegrateSegments(segments_ptrs);
                SemanticSLAM.ClearMapperSegsCache();
            }
            auto semantic_mapping_end = std::chrono::system_clock::now();
            semantic_time += std::chrono::duration<double>(
                semantic_mapping_end-semantic_mapping_start);
           
            use_semantics = true;
        }
        else
        {
            auto track_start = std::chrono::system_clock::now();
            SemanticSLAM.TrackRGBD(imRGB,imD,frame_timestamp);
            auto track_end = std::chrono::system_clock::now();
            localization_time += std::chrono::duration<double>(track_end-track_start);
        }

        if(ni%cout_step == 0)
        {
            std::cout << "Processing time with " << ni << " frames: " << std::endl;
            std::cout << "  total: load + track + semantic" << std::endl;
            std::cout << "  total " << total_run_time.count() << ": " <<  load_time.count() << 
                " + " << localization_time.count() << " + " << semantic_time.count() << std::endl;
        }

        auto frame_end = std::chrono::system_clock::now();
        total_run_time += std::chrono::duration<double>(frame_end-frame_start);
    }

    // Stop all threads
    SemanticSLAM.Shutdown();

    // Save camera trajectory
    SemanticSLAM.SaveTrajectoryTUM(traject_f);
    // Save frame-wise tracking result if necessary
    if (SemanticSLAM.getTrackingDebug())
    {
        std::string orbslam_features_folder = traj_folder + "/features/";
        std::string command = "mkdir -p " + orbslam_features_folder;
        system(command.c_str());
        SemanticSLAM.SaveFeaturesAndMapPointsRGBD(traj_folder);
    }
    // Generate meshes
    if (SemanticSLAM.getSaveMesh())
    {
        std::string orbslam_mesh_folder = result_folder + "/orbslam_mesh";
        std::string command = "mkdir -p " + orbslam_mesh_folder;
        system(command.c_str());
        SemanticSLAM.GenerateMeshes(orbslam_mesh_folder);
    }

    return 0;
}

void LoadImagesPath(const string &sceneFolder, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, int img_num)
{
    vstrImageFilenamesRGB.clear();
    vstrImageFilenamesD.clear();

    std::string depth_folder = sceneFolder + "/depth/";
    std::string rgb_folder = sceneFolder + "/color_warped/";

    int image_index = 0;
    for(image_index = 0; image_index < img_num; image_index++)
    {
        std::string frame_idx_str = std::to_string(image_index);
        // if(image_index==0) cout << "frame_idx_str: " <<   frame_idx_str << endl;
        std::string depth_f = depth_folder + frame_idx_str + ".png";
        std::string rgb_f = rgb_folder + frame_idx_str + ".jpg";

        vstrImageFilenamesD.push_back(depth_f);
        vstrImageFilenamesRGB.push_back(rgb_f);
    }
}
cv::Mat LoadCamera(const string &sceneFolder)
{
    std::array<std::array<float, 4>, 4> K_array;
    std::string intrinsic_file = sceneFolder + "/intrinsic/intrinsic_depth.txt";
    ifstream file(intrinsic_file);
    for (int i = 0; i < 4; ++i) 
        for (int j = 0; j < 4; ++j)
        {file >> K_array[i][j];}
            
    cv::Mat K_mat = (cv::Mat_<float>(3,3) << K_array[0][0], K_array[0][1],K_array[0][2],
                    K_array[1][0], K_array[1][1],K_array[1][2],
                    K_array[2][0], K_array[2][1],K_array[2][2]);
    
    return K_mat;
}


string zfillString(const string &string_to_fill, std::size_t fill_num)
{
    if(string_to_fill.size() > fill_num)
        return string_to_fill;
    
    int precision = fill_num - std::min(fill_num, string_to_fill.size());
    std::string string_filled = std::string(precision, '0').append(string_to_fill);
    return string_filled;
}

void LoadSegmentsPath(const string &segmentFolder, vector<string> &vstrSegInfoFiles,
                vector<string> &vstrSegMaskFiles, int img_num)
{
    vstrSegInfoFiles.clear();
    vstrSegMaskFiles.clear();
    for(int segs_frame_index = 0; segs_frame_index < img_num; segs_frame_index++)
    {
        std::string frame_idx_str = zfillString(std::to_string(segs_frame_index), 5);
        std::string seg_info_file = segmentFolder + "/" + frame_idx_str + "_seg_info.h5";
        std::string seg_mask_file = segmentFolder + "/" + frame_idx_str + "_mask.png";

        vstrSegInfoFiles.push_back(seg_info_file);
        vstrSegMaskFiles.push_back(seg_mask_file);
    }
}

bool LoadSegments(const string &mask_file, const string &segs_info_file, 
    const string &depth_file, const cv::Mat &K, cv::Mat &segments_mask, 
    map<uint8_t, uint8_t>& segs_semantic_map, 
    std::vector<voxblox::SegmentConfidence*> & segments)
{
    // check file exists
    if( (!std::experimental::filesystem::exists(mask_file)) || 
        (!std::experimental::filesystem::exists(segs_info_file)) )
    {
        // std::cout << "  seg files donot exist! " << std::endl;
        return false;
    }
        
    // get segs mask 
    segments_mask = cv::imread(mask_file, cv::IMREAD_UNCHANGED);

    // read segs info from segs_info_file
    segments.clear();
    segs_semantic_map.clear();
    using namespace HighFive;
    File hdf_file(segs_info_file, File::ReadOnly);

    // get num of segs
    auto dataset_segs_num = hdf_file.getDataSet("seg_num");
    int segs_num = dataset_segs_num.read<int>();
    // std::cout << "  segs_num: " << segs_num << std::endl;
    if(segs_num <= 0) return false;

    // get segments information
    auto dataset_is_thing = hdf_file.getDataSet("is_thing");
    std::vector<bool> is_thing_list = dataset_is_thing.read<std::vector<bool>>();
    if(is_thing_list.size() != segs_num) return false;

    auto dataset_instance_labels_float = hdf_file.getDataSet("instance_label");
    std::vector<float> instance_labels_float = dataset_instance_labels_float.read<std::vector<float>>();
    if(instance_labels_float.size() != segs_num) return false;

    auto dataset_inst_confidence = hdf_file.getDataSet("inst_confidence");
    std::vector<float> inst_confidence_list = dataset_inst_confidence.read<std::vector<float>>();
    if(inst_confidence_list.size() != segs_num) return false;

    auto dataset_semantic_labels = hdf_file.getDataSet("class_label");
    std::vector<uint8_t> semantic_labels = dataset_semantic_labels.read<std::vector<uint8_t>>();
    if(semantic_labels.size() != segs_num) return false;

    auto dataset_overlap_ratio = hdf_file.getDataSet("overlap_ratio");
    std::vector<float> overlap_ratio_list = dataset_overlap_ratio.read<std::vector<float>>();
    if(overlap_ratio_list.size() != segs_num) return false;

    auto dataset_poses_eles = hdf_file.getDataSet("pose");
    std::vector<std::vector<std::vector<float>>> poses_eles = 
        dataset_poses_eles.read<std::vector<std::vector<std::vector<float>>>>();
    if(poses_eles.size() != segs_num) return false;

    auto dataset_seg_centers_eles = hdf_file.getDataSet("center");
    std::vector<std::vector<std::vector<float>>> seg_centers_eles = 
        dataset_seg_centers_eles.read<std::vector<std::vector<std::vector<float>>>>();
    if(seg_centers_eles.size() != segs_num) return false;

    // get segments
    cv::Mat depth_img = cv::imread(depth_file, cv::IMREAD_UNCHANGED);
    cv::Mat depth_scaled = cv::Mat::zeros(depth_img.size(), CV_32FC1);
    cv::rgbd::rescaleDepth(depth_img, CV_32FC1, depth_scaled);
    for(uint8_t seg_i=1; seg_i<=segs_num; seg_i++ )
    {
        // get seg points
        cv::Mat seg_points;
        cv::rgbd::depthTo3d(depth_scaled, K, seg_points, (segments_mask==seg_i));
        // std::cout << "  size of seg " << seg_i << "'s points: " << seg_points.rows
        //     << " x " << seg_points.cols <<
        //     "   with type " << seg_points.type() << std::endl;
        // get seg pose
        std::vector<std::vector<float>> pose_vector = poses_eles[seg_i-1];
        Eigen::Matrix<float, 4, 4> T_G_C_eigen;
        T_G_C_eigen << 
            pose_vector[0][0], pose_vector[0][1],pose_vector[0][2],pose_vector[0][3], 
            pose_vector[1][0], pose_vector[1][1],pose_vector[1][2],pose_vector[1][3],
            pose_vector[2][0], pose_vector[2][1],pose_vector[2][2],pose_vector[2][3],
            pose_vector[3][0], pose_vector[3][1],pose_vector[3][2],pose_vector[3][3];
        Transformation T_G_C(T_G_C_eigen);
        // get seg center
        std::vector<std::vector<float>> seg_center = seg_centers_eles[seg_i-1];
        // set center as b_box  
        cv::Mat b_box = (cv::Mat_<float>(1,3) << seg_center[0][0], seg_center[0][1], seg_center[0][2]);
        // get instance and semantic info
        bool is_instance = is_thing_list[seg_i-1];
        std::uint16_t instance_label = std::uint16_t(instance_labels_float[seg_i-1]);
        float inst_confidence = inst_confidence_list[seg_i-1];
        std::uint8_t semantic_label = semantic_labels[seg_i-1];
        float overlap_ratio = overlap_ratio_list[seg_i-1];

        // create segment and save ptr
        SegmentConfidence* segment_ptr = 
            new SegmentConfidence(
                &seg_points, &b_box, instance_label, semantic_label,
                T_G_C, inst_confidence, overlap_ratio, is_instance);
        segments.push_back( segment_ptr );

        // save segment_idx to semantic map
        segs_semantic_map.emplace(seg_i, semantic_label);
    }
    return true;
}

void TestLoadSegments(vector<string> &vstrSegInfoFiles,vector<string> &vstrSegMaskFiles,
    vector<string> &vstrImageFilenamesD, const cv::Mat &K, int img_idx)
{
    std::vector<voxblox::SegmentConfidence*> test_segments;
    cv::Mat segments_mask;
    std::map<uint8_t, uint8_t> segs_semantics_map;
    
    std::cout << "Test Loading ... Image idx " << img_idx << std::endl;
    std::cout << "Loading file " << vstrSegInfoFiles[img_idx] << "... " << std::endl;
    std::cout << "Loading file " << vstrSegMaskFiles[img_idx] << "... " << std::endl;
    std::cout << "Loading file " << vstrImageFilenamesD[img_idx] << "... " << std::endl;
    std::cout << "Camera Model: " << std::endl << 
        K.at<float>(0,0) << ", " << K.at<float>(0,1) << ", " << K.at<float>(0,2) << std::endl <<
        K.at<float>(1,0) << ", " << K.at<float>(1,1) << ", " << K.at<float>(1,2) << std::endl <<
        K.at<float>(2,0) << ", " << K.at<float>(2,1) << ", " << K.at<float>(2,2) << std::endl;
    LoadSegments(vstrSegMaskFiles[img_idx], vstrSegInfoFiles[img_idx], 
        vstrImageFilenamesD[img_idx], K, segments_mask, segs_semantics_map, 
        test_segments);
    
    std::cout << "Get " << int(test_segments.size()) << " segments" << std::endl;
    int segment_idx = 0;
    for(voxblox::SegmentConfidence* & segment_ptr: test_segments)
    {
        std::cout << "Segment " << segment_idx << std::endl;
        std::cout << "  number of points " << int(segment_ptr->points_C_.size()) << std::endl;
        std::cout << "  first points " << segment_ptr->points_C_[0].transpose() << std::endl;
        std::cout << "  last points " << segment_ptr->points_C_.back().transpose() << std::endl;
        std::cout << "  center points " << segment_ptr->b_box_.at<float>(0,0) << ", "
             << segment_ptr->b_box_.at<float>(0,1) << ", " 
             << segment_ptr->b_box_.at<float>(0,2) << std::endl;
        std::cout << "  inst label " << int(segment_ptr->instance_label_) << std::endl;
        std::cout << "  inst_confidence " << (segment_ptr->inst_confidence_) << std::endl;
        std::cout << "  semantic label " << int(segment_ptr->semantic_label_) << std::endl;
        std::cout << "  obj_seg_confidence: " << (segment_ptr->obj_seg_confidence_) << std::endl;
        std::cout << "  is thing: " << (segment_ptr->is_thing_) << std::endl;
        std::cout << "  pose: " << (segment_ptr->T_G_C_) << std::endl;
        segment_idx++;
    }
    std::cout << "segs_semantics_map: " << std::endl;
    for(std::map<uint8_t, uint8_t>::iterator seg_semantic_it = segs_semantics_map.begin();
        seg_semantic_it != segs_semantics_map.end(); seg_semantic_it++ )
    {
        std::cout << "segs idx: " << int(seg_semantic_it->first) << " with semantics " 
            << int(seg_semantic_it->second) << std::endl;
    }

}
