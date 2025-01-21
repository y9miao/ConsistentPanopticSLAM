#include "SemanticAided/SystemSemantic.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <openssl/md5.h>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

using namespace std;

namespace ORB_SLAM3
{

    SystemSemantic::SystemSemantic(
        const string &strVocFile, 
        const string &ORBSlamSettingsFile, 
        const string &SemanticMappingSettingsFile, 
        const eSensor sensor, 
        const bool bUseViewer, 
        const int initFr, 
        const string &strSequence):
        System(strVocFile, ORBSlamSettingsFile, sensor, 
            bUseViewer, initFr, strSequence, true)
    {
        // Voxel-TSDF semantic-instance mapper 
        gsm_mapper_ptr = std::make_shared<ConsistentGSMMapper>(SemanticMappingSettingsFile);

        //Check semantic settings file
        cv::FileStorage fsSettings_semantic(SemanticMappingSettingsFile.c_str(), cv::FileStorage::READ);
        cv::FileNode semantic_node = fsSettings_semantic["Result.save_meshes"];
        if(!semantic_node.empty()){
            save_mesh_ = ( (std::string)semantic_node != "false" );
            std::cout << "Save Mesh: " << save_mesh_ << std::endl;
        }
        //Check orbslam settings file
        cv::FileStorage fsSettings(ORBSlamSettingsFile.c_str(), cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << ORBSlamSettingsFile << endl;
            exit(-1);
        }
        cv::FileNode node = fsSettings["Tracking.mode"];
        if(!node.empty()){
            tracking_mode_ = (int)node;
            std::cout << "Tracking Mode: " << tracking_mode_ << std::endl;
        }
        node = fsSettings["Tracking.debug"];
        if(!node.empty()){
            tracking_debug_ = ( (std::string)node != "false" );
            std::cout << "Tracking Debug: " << tracking_debug_ << std::endl;
        }

    }

    Sophus::SE3f SystemSemantic::TrackRGBDSemantic(
    const cv::Mat &im, 
    const cv::Mat &depthmap, 
    const cv::Mat &segment_mask,
    const map<uint8_t, uint8_t> &segs_semantic_map,
    const double &timestamp, 
    string filename)
    {
        // make sure the right sensor
        if(mSensor!=RGBD)
        {
            cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
            exit(-1);
        }
        // resize the image if necessary
        cv::Mat imToFeed = im.clone();
        cv::Mat imDepthToFeed = depthmap.clone();
        if(settings_ && settings_->needToResize()){
            cv::Mat resizedIm;
            cv::resize(im,resizedIm,settings_->newImSize());
            imToFeed = resizedIm;
            cv::resize(depthmap,imDepthToFeed,settings_->newImSize());
        }
        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if(mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while(!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if(mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if(mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
                mbResetActiveMap = false;
            }
            else if(mbResetActiveMap)
            {
                mpTracker->ResetActiveMap();
                mbResetActiveMap = false;
            }
        }
        // Track with frame
        Sophus::SE3f Tcw;
        Tcw = dynamic_cast<TrackingRGBDSemantics*>(mpTracker)->GrabImageRGBDSemanticsOriginal(
            imToFeed,imDepthToFeed,timestamp, segment_mask, segs_semantic_map);
        // if(tracking_mode_ != 0)
        // {
        //     Tcw = dynamic_cast<TrackingRGBDSemantics*>(mpTracker)->GrabImageRGBDSemanticsOriginal(
        //         imToFeed,imDepthToFeed,timestamp, segment_mask, segs_semantic_map);
        // }
        // else
        // {
        //     Tcw = mpTracker->GrabImageRGBD(imToFeed,imDepthToFeed,timestamp,filename);
        // }

        // update current state and frame information
        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        return Tcw;
    }

    void SystemSemantic::IntegrateSegments(std::vector<SegmentConfidence*>& segments_ptrs)
    {
        gsm_mapper_ptr->integrateSegments(segments_ptrs);
    }
    void SystemSemantic::ClearMapperSegsCache()
    {
        gsm_mapper_ptr->clearFrameSegsCache();
    }
    void SystemSemantic::GenerateMeshes(const string& mesh_folder)
    {
        gsm_mapper_ptr->generateMesh(mesh_folder);   
    }

    void SystemSemantic::GetSemanticsWithCoords(
        std::vector<voxblox::Point> query_pcl,
        std::vector<SemanticLabel>& semantics_list)
    {
        (gsm_mapper_ptr.get())->quarySemanticPointCloud(query_pcl,semantics_list);
    }
    voxblox::Label SystemSemantic::GetSuperpointIDWithCoord(voxblox::Point query_point)
    {
        return (gsm_mapper_ptr.get())->quarySuperpointID(query_point);
    }
}