#ifndef TRACKING_RGBD_SEMANTICS_H
#define TRACKING_RGBD_SEMANTICS_H
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
// orbslam
#include "Tracking.h" 
#include "MapPoint.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

// semantics
#include "consistent_mapping/consistent_gsm_mapper.h"

namespace ORB_SLAM3
{

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;
class Tracking;

class TrackingRGBDSemantics: public Tracking
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TrackingRGBDSemantics(
        System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, 
        MapDrawer* pMapDrawer, Atlas* pAtlas, KeyFrameDatabase* pKFDB, 
        const string &strSettingPath, const int sensor, Settings* settings, 
        const string &_nameSeq=std::string());

    Sophus::SE3f GrabImageRGBDSemanticsOriginal(
        const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp,
        const cv::Mat &segment_mask, const map<uint8_t, uint8_t> &segs_semantic_map
    );

    void TrackSemanticsOriginal(
        const cv::Mat &segment_mask, 
        const map<uint8_t, uint8_t> &segs_semantic_map);

    int FilterFeatureMatchesSemantics(Frame* frame_ptr, const cv::Mat &segment_mask,
        const map<uint8_t, uint8_t> &segs_semantic_map);
    int FilterFeatureMatchesSuperpoints(Frame* frame_ptr, const cv::Mat &segment_mask);

    bool TrackReferenceKeyFrameSemantics(
        const cv::Mat &segment_mask, 
        const map<uint8_t, uint8_t> &segs_semantic_map);
    bool TrackWithMotionModelSemantics(
        const cv::Mat &segment_mask, 
        const map<uint8_t, uint8_t> &segs_semantic_map);
    Settings* settings_;
    int tracking_mode_ = 0;
    bool tracking_debug_ = false;
    
};

}

#endif // TRACKING_RGBD_SEMANTICS_H