#include "SemanticAided/SystemSemantic.h"
#include "SemanticAided/TrackingRGBDSemantics.h"

namespace ORB_SLAM3
{

    TrackingRGBDSemantics::TrackingRGBDSemantics(
        System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, 
        MapDrawer* pMapDrawer, Atlas* pAtlas, KeyFrameDatabase* pKFDB, 
        const string &strSettingPath, const int sensor, Settings* settings, 
        const string &_nameSeq):
        Tracking(pSys, pVoc, pFrameDrawer, pMapDrawer, pAtlas, 
            pKFDB, strSettingPath, sensor, settings, _nameSeq),
        settings_(settings)
    {
        tracking_mode_ = settings->trackMode();
        tracking_debug_ = settings->trackDebug();
    }
    
    Sophus::SE3f TrackingRGBDSemantics::GrabImageRGBDSemanticsOriginal(
        const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp,
        const cv::Mat &segment_mask, const map<uint8_t, uint8_t> &segs_semantic_map
    )
    {
        // preprocess images
        mImGray = imRGB;
        cv::Mat imDepth = imD;

        if(mImGray.channels()==3)
        {
            if(mbRGB)
                cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            else
                cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
        }
        else if(mImGray.channels()==4)
        {
            if(mbRGB)
                cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            else
                cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
        }
        if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
            imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

        // generate frame and extrack Orb features
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,
            mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
        mCurrentFrame.mNameFile = "";
        mCurrentFrame.mnDataset = mnNumDataset;
        TrackSemanticsOriginal(segment_mask, segs_semantic_map);
        return mCurrentFrame.GetPose();
    }

    void TrackingRGBDSemantics::TrackSemanticsOriginal(
        const cv::Mat &segment_mask, 
        const map<uint8_t, uint8_t> &segs_semantic_map
    )
    {
        if (bStepByStep)
        {
            std::cout << "Tracking: Waiting to the next step" << std::endl;
            while(!mbStep && bStepByStep)
                usleep(500);
            mbStep = false;
        }
        Map* pCurrentMap = mpAtlas->GetCurrentMap();
        if(!pCurrentMap)
        { cout << "ERROR: There is not an active map in the atlas" << endl; }
        if(mState!=NO_IMAGES_YET)
        {
            // reset if timestamps are not correct
            if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
            {
                cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
                unique_lock<mutex> lock(mMutexImuQueue);
                mlQueueImuData.clear();
                CreateMapInAtlas();
                return;
            }
            else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+2.0)
            {
                // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
                // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
                if(mpAtlas->isInertial())
                {

                    if(mpAtlas->isImuInitialized())
                    {
                        cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                        if(!pCurrentMap->GetIniertialBA2())
                        {
                            mpSystem->ResetActiveMap();
                        }
                        else
                        {
                            CreateMapInAtlas();
                        }
                    }
                    else
                    {
                        cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                        mpSystem->ResetActiveMap();
                    }
                    return;
                }

            }
        }
        if(mState==NO_IMAGES_YET)
        { mState = NOT_INITIALIZED; }

        mLastProcessedState=mState;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);
        mbMapUpdated = false;
        int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
        int nMapChangeIndex = pCurrentMap->GetLastMapChange();
        if(nCurMapChangeIndex>nMapChangeIndex)
        {
            pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
            mbMapUpdated = true;
        }
        
        if(mState==NOT_INITIALIZED) // initialize
        { 
            StereoInitialization();
            if(mState!=OK) // If rightly initialized, mState=OK
            {
                mLastFrame = Frame(mCurrentFrame);
                return;
            }
            if(mpAtlas->GetAllMaps().size() == 1)
            {
                mnFirstFrameId = mCurrentFrame.mnId;
            }
        }
        else // Track frame
        {
            bool bOK;

            if(mState==OK)  // if state is good
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();
                // track with reference frame or motion model
                if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackReferenceKeyFrameSemantics(segment_mask, segs_semantic_map);
                }
                else
                {
                    Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackWithMotionModelSemantics(segment_mask, segs_semantic_map);
                    if(!bOK)
                        bOK = TrackReferenceKeyFrameSemantics(segment_mask, segs_semantic_map);
                }

                if (!bOK)
                {
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST;
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10)
                    {
                        // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                    }
                    else
                    {
                        mState = LOST;
                    }
                }
            }
            else // if state is LOST
            {
                if (mState == RECENTLY_LOST)
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);
                    bOK = true;

                    // Relocalization
                    bOK = Relocalization();
                    if(mCurrentFrame.mTimeStamp-mTimeStampLost>3.0f && !bOK)
                    {
                        mState = LOST;
                        Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                        bOK=false;
                    }
                }
                else if (mState == LOST)
                {
                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);
                    if (pCurrentMap->KeyFramesInMap()<10)
                    {
                        mpSystem->ResetActiveMap();
                        Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    }else
                        CreateMapInAtlas();

                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
                    return;
                }
            }



            if(!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if(bOK)
            {
                bOK = TrackLocalMap();

            }
            if(!bOK)
                cout << "Fail to track local map!" << endl;

            // adjust state according to tracking result
            if(bOK)
                mState = OK;
            else if (mState == OK)
            {
                mState=RECENTLY_LOST;
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            }

            // Update drawer
            mpFrameDrawer->Update(this);
            if(mCurrentFrame.isSet())
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

            if(bOK || mState==RECENTLY_LOST)
            {
                // Update motion model
                if(mLastFrame.isSet() && mCurrentFrame.isSet())
                {
                    Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                    mVelocity = mCurrentFrame.GetPose() * LastTwc;
                    mbVelocity = true;
                }
                else {
                    mbVelocity = false;
                }
                // Clean VO matches
                for(int i=0; i<mCurrentFrame.N; i++)
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                    if(pMP)
                        if(pMP->Observations()<1)
                        {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                        }
                }
                // Delete temporal MapPoints
                for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
                {
                    MapPoint* pMP = *lit;
                    delete pMP;
                }
                mlpTemporalPoints.clear();

                // add new keyframe if needed
                bool bNeedKF = NeedNewKeyFrame();
                if(bNeedKF && bOK)
                    CreateNewKeyFrame();

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame. Only has effect if lastframe is tracked
                for(int i=0; i<mCurrentFrame.N;i++)
                {
                    if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            if(mState==LOST)
            {
                if(pCurrentMap->KeyFramesInMap()<=10)
                {
                    mpSystem->ResetActiveMap();
                    return;
                }

                CreateMapInAtlas();
                return;
            }

            if(!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;
            mLastFrame = Frame(mCurrentFrame);
        }
        
        if(mState==OK || mState==RECENTLY_LOST)
        {
            // Store frame pose information to retrieve the complete camera trajectory afterwards.
            if(mCurrentFrame.isSet())
            {
                Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
                mlRelativeFramePoses.push_back(Tcr_);
                mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
                mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                mlbLost.push_back(mState==LOST);
            }
            else
            {
                // This can happen if tracking is lost
                mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                mlpReferences.push_back(mlpReferences.back());
                mlFrameTimes.push_back(mlFrameTimes.back());
                mlbLost.push_back(mState==LOST);
            }
        }


    }

    int TrackingRGBDSemantics::FilterFeatureMatchesSemantics(
        Frame* frame_ptr, const cv::Mat &segment_mask,
        const map<uint8_t, uint8_t> &segs_semantic_map)
    {
        std::vector<int> pairs_idxs_list;
        std::vector<cv::KeyPoint> feature_locs_list;
        std::vector<voxblox::Point> map_points_list;
        std::vector<int> incosistent_pairs_idxs_list;
        
        pairs_idxs_list.reserve(frame_ptr->N);
        feature_locs_list.reserve(frame_ptr->N);
        map_points_list.reserve(frame_ptr->N);
        incosistent_pairs_idxs_list.reserve(frame_ptr->N);

        // std::cout << "  Start filtering out features with semantics"
        //     << std::endl;

        //  get 3D-2D paries
        for(int pair_idx=0; pair_idx<frame_ptr->N; pair_idx++)
        {
            if(frame_ptr->mvpMapPoints[pair_idx])
            {
                pairs_idxs_list.push_back(pair_idx);
                feature_locs_list.push_back(frame_ptr->mvKeysUn[pair_idx]);

                MapPoint* pMP = frame_ptr->mvpMapPoints[pair_idx];
                map_points_list.push_back(
                    voxblox::Point(pMP->GetWorldPos()) );
            }
        }
        // std::cout << "  3D-2D pairs: " << pairs_idxs_list.size()
        //     << std::endl;

        // get semantics of 3D mapPoints
        std::vector<voxblox::SemanticLabel> semantics_points_list;
        std::vector<voxblox::SemanticLabel> semantics_features_list;
        dynamic_cast<SystemSemantic*>(mpSystem)->GetSemanticsWithCoords(
            map_points_list, semantics_points_list);
        // get semantics of 2D points
        for(cv::KeyPoint& feature_loc:feature_locs_list)
        {
            int loc_y = feature_loc.pt.y;
            int loc_x = feature_loc.pt.x;
            uint8_t seg_label = segment_mask.at<uint8_t>(loc_y, loc_x);
            std::map<uint8_t, uint8_t>::const_iterator seg_semantic_it = 
                segs_semantic_map.find(seg_label);
            if(seg_label==0 || seg_semantic_it == segs_semantic_map.end())
                semantics_features_list.push_back(voxblox::BackgroundSemLabel);
            else
                semantics_features_list.push_back(seg_semantic_it->second);
            
        }
        // check semantic consistent 
        CHECK_EQ(semantics_points_list.size(), semantics_features_list.size());
        CHECK_EQ(semantics_points_list.size(), pairs_idxs_list.size());
        for(int i = 0; i<pairs_idxs_list.size(); i++)
        {
            bool is_point_NOT_background = 
                (semantics_points_list[i] != voxblox::BackgroundSemLabel);
            bool is_feature_NOT_background = 
                (semantics_features_list[i] != voxblox::BackgroundSemLabel);
            bool is_semantic_NOT_consistent = 
                (semantics_points_list[i]  != semantics_features_list[i]);

            if(is_point_NOT_background && is_feature_NOT_background
                && is_semantic_NOT_consistent)
            { // record 3D-2D pairs with inconsistent semantics
                incosistent_pairs_idxs_list.push_back(pairs_idxs_list[i]);
            }
        }
        // clear pairs with inconsistent semantics
        for(int inconsistent_pair_idx:incosistent_pairs_idxs_list)
        {
            frame_ptr->mvpMapPoints[inconsistent_pair_idx] = 
                static_cast<MapPoint*>(NULL);
        }
        // std::cout << "  Removing " << incosistent_pairs_idxs_list.size() << 
        //     " pairs from " << pairs_idxs_list.size() << " pairs" << std::endl;

        return pairs_idxs_list.size() - incosistent_pairs_idxs_list.size();
    }

    int TrackingRGBDSemantics::FilterFeatureMatchesSuperpoints(
        Frame* frame_ptr, const cv::Mat &segment_mask)
    {
        std::vector<int> pairs_idxs_list;
        std::vector<cv::KeyPoint> feature_locs_list;
        std::vector<voxblox::Point> map_points_list;
        std::vector<int> incosistent_pairs_idxs_list;
        std::map<uint8_t, std::vector<int>> map_seg_id__pairs_idxs;
        
        pairs_idxs_list.reserve(frame_ptr->N);
        feature_locs_list.reserve(frame_ptr->N);
        map_points_list.reserve(frame_ptr->N);
        incosistent_pairs_idxs_list.reserve(frame_ptr->N);

        //  get 3D-2D paries
        for(int pair_idx=0; pair_idx<frame_ptr->N; pair_idx++)
        {
            if(frame_ptr->mvpMapPoints[pair_idx])
            {
                pairs_idxs_list.push_back(pair_idx);
                feature_locs_list.push_back(frame_ptr->mvKeysUn[pair_idx]);

                MapPoint* pMP = frame_ptr->mvpMapPoints[pair_idx];
                map_points_list.push_back(
                    voxblox::Point(pMP->GetWorldPos()) );

                // get all 3D-2D pairs for each segments(kps located in seg region)
                int loc_y = frame_ptr->mvKeysUn[pair_idx].pt.y;
                int loc_x = frame_ptr->mvKeysUn[pair_idx].pt.x;
                uint8_t seg_label = segment_mask.at<uint8_t>(loc_y, loc_x);
                if(seg_label != 0)
                {
                    std::map<uint8_t, std::vector<int>>::iterator segs_id__pairs_it = 
                        map_seg_id__pairs_idxs.find(seg_label);
                    if(segs_id__pairs_it != map_seg_id__pairs_idxs.end())
                    {
                        segs_id__pairs_it->second.push_back(pair_idx);
                    }
                    else
                    {
                        map_seg_id__pairs_idxs.emplace(
                            seg_label, std::vector<int>({pair_idx}));
                    }
                }
            }
        }

        // determine super-point id for each seg and filter out pairs with inconsistent super-point id 
        for(std::map<uint8_t, std::vector<int>>::iterator segs_id__pairs_it = map_seg_id__pairs_idxs.begin();
            segs_id__pairs_it != map_seg_id__pairs_idxs.end(); segs_id__pairs_it++)
        {
            // calculate vote for each super point id 
            std::map<uint16_t, int> map_superpoint_count;
            std::map<int, uint16_t> map_pair_idx__superpoint;
            for(int pair_idx: segs_id__pairs_it->second)
            {
                MapPoint* pMP = frame_ptr->mvpMapPoints[pair_idx];
                voxblox::Label superpoint_id = 
                    dynamic_cast<SystemSemantic*>(mpSystem)->GetSuperpointIDWithCoord(
                        voxblox::Point(pMP->GetWorldPos()));
                map_pair_idx__superpoint.emplace(pair_idx, superpoint_id);

                std::map<uint16_t, int>::iterator superpoint_count_it = 
                    map_superpoint_count.find(superpoint_id);
                if(superpoint_count_it != map_superpoint_count.end())
                    superpoint_count_it->second ++ ;
                else
                    map_superpoint_count.emplace(superpoint_id, 1);
            }
            // get super point id with most vote
            int max_vote = 0;
            voxblox::Label superid_voted = voxblox::BackgroundLabel;
            for(std::map<uint16_t, int>::iterator superpoint_count_it = map_superpoint_count.begin();
                superpoint_count_it != map_superpoint_count.end(); superpoint_count_it++)
            {
                if(superpoint_count_it->second > max_vote)
                    superid_voted = superpoint_count_it->first;
            }
            // filter out pairs with inconsistent pairs 
            if(superid_voted != voxblox::BackgroundLabel)
            {
                for(int pair_idx: segs_id__pairs_it->second)
                {
                    std::map<int, uint16_t>::iterator pair_superpoint_it = 
                        map_pair_idx__superpoint.find(pair_idx);
                    if(pair_superpoint_it != map_pair_idx__superpoint.end())
                    {
                        bool is_point_NOT_background = 
                            (pair_superpoint_it->second != voxblox::BackgroundLabel);
                        bool is_NOT_consistent = 
                            (pair_superpoint_it->second != superid_voted);
                        if(is_point_NOT_background && is_NOT_consistent)
                            incosistent_pairs_idxs_list.push_back(pair_idx);
                    }
                }
            } 
        }
            
        // clear pairs with inconsistent semantics
        for(int inconsistent_pair_idx:incosistent_pairs_idxs_list)
        {
            frame_ptr->mvpMapPoints[inconsistent_pair_idx] = 
                static_cast<MapPoint*>(NULL);
        }
        // std::cout << "  Removing " << incosistent_pairs_idxs_list.size() << 
        //     " pairs from " << pairs_idxs_list.size() << " pairs" << std::endl;

        return pairs_idxs_list.size() - incosistent_pairs_idxs_list.size();
    }


    bool TrackingRGBDSemantics::TrackReferenceKeyFrameSemantics(
        const cv::Mat &segment_mask, 
        const map<uint8_t, uint8_t> &segs_semantic_map
    )
    {
        // std::cout << "  Tracking With reference Frame !!! " << std::endl;
        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7,true);
        vector<MapPoint*> vpMapPointMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
        mCurrentFrame.mvpMapPoints = vpMapPointMatches;

        // filter out 3D-2D pairs with different semantics 
        if(tracking_mode_ == 1 && segs_semantic_map.size()>0)
        {
            nmatches = FilterFeatureMatchesSemantics(&mCurrentFrame, segment_mask,segs_semantic_map);
        }
        if(tracking_mode_ == 2 && segs_semantic_map.size()>0)
        {
            nmatches = FilterFeatureMatchesSuperpoints(&mCurrentFrame, segment_mask);
        }
        // store frame-wise information for debug
        if(tracking_debug_)
        {
            frame_ids_list.push_back(int(round( mCurrentFrame.mTimeStamp*settings_->fps() ) ));
            keypoints_list.push_back(mCurrentFrame.mvKeysUn);
            mvpMapPoints_list.push_back(mCurrentFrame.mvpMapPoints);
            mvuRight_list.push_back(mCurrentFrame.mvuRight);
            // save temporal 3D points' location
            std::map<MapPoint*, Eigen::Vector3f> mvpMapPointsTempLoc;
            for(MapPoint* & map_point_ptr:mCurrentFrame.mvpMapPoints)
            {
                if(map_point_ptr)
                {
                    mvpMapPointsTempLoc.emplace(map_point_ptr, map_point_ptr->GetWorldPos());
                }
            } 
            mvpMapPointsTempLocs_list.push_back(mvpMapPointsTempLoc);
            mvpMapPointsRes_list.push_back(mCurrentFrame.mapPoints_residual_);
            frames_temp_poses.emplace(mCurrentFrame.mTimeStamp,mCurrentFrame.GetPose());
        }
        if(nmatches<15)
        {
            cout << "TRACK_REF_KF: Less than 15 matches!!\n";
            return false;
        }
        
        mCurrentFrame.SetPose(mLastFrame.GetPose());
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            //if(i >= mCurrentFrame.Nleft) break;
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i]=false;
                    if(i < mCurrentFrame.Nleft){
                        pMP->mbTrackInView = false;
                    }
                    else{
                        pMP->mbTrackInViewR = false;
                    }
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    nmatchesMap++;
            }
        }

        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return nmatchesMap>=10;
    }

    bool TrackingRGBDSemantics::TrackWithMotionModelSemantics(
            const cv::Mat &segment_mask, 
            const map<uint8_t, uint8_t> &segs_semantic_map
    )
    {
        // std::cout << "  Tracking With Motion Model !!! " << std::endl;
        ORBmatcher matcher(0.9,true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
        {
            // Predict state with IMU if it is initialized and it doesnt need reset
            PredictStateIMU();
            return true;
        }
        else
        {
            mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
        }

        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        // Project points seen in previous frame
        int th;

        if(mSensor==System::STEREO)
            th=7;
        else
            th=15;

        int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        if(tracking_mode_ == 1 && segs_semantic_map.size()>0)
        {
            nmatches = FilterFeatureMatchesSemantics(&mCurrentFrame, segment_mask,segs_semantic_map);
        }
        if(tracking_mode_ == 2 && segs_semantic_map.size()>0)
        {
            nmatches = FilterFeatureMatchesSuperpoints(&mCurrentFrame, segment_mask);
        }
        // store frame-wise information for debug
        if(tracking_debug_)
        {
            frame_ids_list.push_back(int(round( mCurrentFrame.mTimeStamp*settings_->fps() ) ));
            keypoints_list.push_back(mCurrentFrame.mvKeysUn);
            mvpMapPoints_list.push_back(mCurrentFrame.mvpMapPoints);
            mvuRight_list.push_back(mCurrentFrame.mvuRight);
            // save temporal 3D points' location
            std::map<MapPoint*, Eigen::Vector3f> mvpMapPointsTempLoc;
            for(MapPoint* & map_point_ptr:mCurrentFrame.mvpMapPoints)
            {
                if(map_point_ptr)
                {
                    mvpMapPointsTempLoc.emplace(map_point_ptr, map_point_ptr->GetWorldPos());
                }
            } 
            mvpMapPointsTempLocs_list.push_back(mvpMapPointsTempLoc);
            mvpMapPointsRes_list.push_back(mCurrentFrame.mapPoints_residual_);
            frames_temp_poses.emplace(mCurrentFrame.mTimeStamp,mCurrentFrame.GetPose());
        }
        // If few matches, uses a wider window search
        if(nmatches<20)
        {
            Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

            nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
            Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

        }

        if(nmatches<20)
        {
            Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                return true;
            else
                return false;
        }

        // Optimize frame pose with all matches
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i]=false;
                    if(i < mCurrentFrame.Nleft){
                        pMP->mbTrackInView = false;
                    }
                    else{
                        pMP->mbTrackInViewR = false;
                    }
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    nmatchesMap++;
            }
        }

        if(mbOnlyTracking)
        {
            mbVO = nmatchesMap<10;
            return nmatches>20;
        }

        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return nmatchesMap>=10;
    }


}