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
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>
#include<System.h>

#define COMPILEDWITHC11

using namespace std;

void LoadImagesPath(const string &sceneFolder, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, int img_num);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_scannet path_to_vocabulary settings_file scenefolder img_num" << endl;
        return 1;
    }

    string vocab_f = string(argv[1]);
    string settings_f = string(argv[2]);
    string scenefolder = string(argv[3]);
    int img_num = std::stoi(argv[4]);

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    LoadImagesPath(scenefolder, vstrImageFilenamesRGB, vstrImageFilenamesD, img_num);

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
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,false);
    float imageScale = SLAM.GetImageScale();
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
    for(int ni=0; ni<nImages; ni++)
    {
        std::cout << "Processing frame id " << ni << std::endl;
        // Read image and depthmap from file
        imRGB = cv::imread(vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imD = cv::imread(vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        // double tframe = vTimestamps[ni];
        double tframe = double(ni);

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe);


        // vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        // double T=0;
        // if(ni<nImages-1)
        //     T = vTimestamps[ni+1]-tframe;
        // else if(ni>0)
        //     T = tframe-vTimestamps[ni-1];

        // if(ttrack<T)
        //     usleep((T-ttrack)*1e6);

        // // pause in somes frames
        // int frame_pause = 0;
        // std::string wait_key;
        // if(ni == frame_pause)
        // {
        //     std::cout << "pause at frame " << frame_pause << " , waiting to step." << std::endl;
        //     std::cin >> wait_key;
        // }
            
    }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    cout << "-------" << endl << endl;
    cout << "using time: " << ttrack << endl;

    // Save camera trajectory
    string traject_f = scenefolder + "/orbslam/trajectory_orb_slam.txt";
    SLAM.SaveTrajectoryTUM(traject_f);
    // SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   
    std:;string result_folder = scenefolder + "/orbslam";
    SLAM.SaveFeaturesAndMapPointsRGBD(result_folder);

    return 0;
}

void LoadImagesPath(const string &sceneFolder, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, int img_num)
{
    bool load_images;

    std::string depth_folder = sceneFolder + "/depth/";
    std::string rgb_folder = sceneFolder + "/rgb/";

    int image_index = 0;
    for(image_index = 0; image_index < img_num; image_index++)
    {
        std::string frame_idx_str = std::to_string(image_index);
        // if(image_index==0) cout << "frame_idx_str: " <<   frame_idx_str << endl;
        std::string depth_f = depth_folder + frame_idx_str + ".png";
        std::string rgb_f = rgb_folder + frame_idx_str + ".png";

        vstrImageFilenamesD.push_back(depth_f);
        vstrImageFilenamesRGB.push_back(rgb_f);
    }
}
