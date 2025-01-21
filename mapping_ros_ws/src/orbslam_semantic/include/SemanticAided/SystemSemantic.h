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


#ifndef SYSTEMSEMANTIC_H
#define SYSTEMSEMANTIC_H


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <map>
#include <opencv2/core/core.hpp>

#include "System.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"
#include "ImuTypes.h"
#include "Settings.h"

#include "SemanticAided/TrackingRGBDSemantics.h"
#include "consistent_mapping/consistent_gsm_mapper.h"



namespace ORB_SLAM3
{

class TrackingRGBDSemantics;

class SystemSemantic: public System
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    SystemSemantic(
        const string &strVocFile, 
        const string &ORBSlamSettingsFile, 
        const string &SemanticMappingSettingsFile, 
        const eSensor sensor, 
        const bool bUseViewer = true, 
        const int initFr = 0, 
        const string &strSequence = std::string());
        
    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackRGBDSemantic(
        const cv::Mat &im, 
        const cv::Mat &depthmap, 
        const cv::Mat &segment_mask,
        const map<uint8_t, uint8_t> &segs_semantic_map,
        const double &timestamp, 
        string filename="");

    void IntegrateSegments(std::vector<SegmentConfidence*>& segments_ptrs);
    void ClearMapperSegsCache();
    void GenerateMeshes(const string& mesh_folder);

    // retrieve tracking state 
    Tracking::eTrackingState getTrackingState()
    { return mpTracker->mState; }
    bool getTrackingDebug() { return tracking_debug_;}
    bool getSaveMesh() { return save_mesh_;}

    void GetSemanticsWithCoords(
        std::vector<voxblox::Point> query_pcl,
        std::vector<SemanticLabel>& semantics_list);
    voxblox::Label GetSuperpointIDWithCoord( voxblox::Point query_point);


private:
    // Map structure that stores 3D Voxel-TSDF-Semantic map
    std::shared_ptr<ConsistentGSMMapper> gsm_mapper_ptr;
    // tracking mode
    int tracking_mode_ = 0;
    bool tracking_debug_ = false;
    bool save_mesh_ = false;
};

}// namespace ORB_SLAM

#endif // SYSTEMSEMANTIC_H
