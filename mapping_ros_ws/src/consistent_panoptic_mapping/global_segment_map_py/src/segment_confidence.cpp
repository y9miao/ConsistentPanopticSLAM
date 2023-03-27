#include "global_segment_map_py/segment_confidence.h"

namespace voxblox {
    bool MY_DEBUG = false;

    SegmentConfidence::SegmentConfidence(
        const cv::Mat* points, // float
        // const cv::Mat* colors, // rgba uint8_t
        // const cv::Mat* geometry_confidence, //float32
        const cv::Mat* b_box, //float32
        InstanceLabel instance_label, //uint16_t
        SemanticLabel semantic_label, //uint8_t
        Transformation& T_G_C,
        ObjSegConfidence inst_confidence,
        ObjSegConfidence obj_seg_confidence,
        bool is_thing):
    Segment(points,instance_label,semantic_label,T_G_C),
    inst_confidence_(inst_confidence),
    obj_seg_confidence_(obj_seg_confidence), 
    is_thing_(is_thing),
    b_box_(*b_box)
    {

        // geometry_confidence_.clear();
        // // size of geometry_confidence should be (num_points, 1)
        // geometry_confidence_.reserve(geometry_confidence->total()); 

        // for(int i=0; i<geometry_confidence->cols; i++)
        // {
        //     geometry_confidence_.push_back(geometry_confidence->at<float>(0,i));
        //     // LOG_EVERY_N(INFO, 10000) << "     PointConfidence: " << geometry_confidence->at<float>(0,i)  
        //     //     << "  ;  location:  " <<  points->at<float>(i,0) << ", "<<  points->at<float>(i,1) << ", "<<  points->at<float>(i,2) ;
        // }
        // LOG(INFO) << "      geometry_confidence_ size: " << geometry_confidence_.size();

        float p_count = 0;
        if(MY_DEBUG){
            // LOG(INFO) << "  Segments newed! ";
            // LOG(INFO) << "      T_G_C: " << T_G_C.log().transpose();
            // LOG(INFO) << "      points rows x cols: " << points->rows << " x " << points->cols;
            // LOG(INFO) << "      points dtype: " << points->type();
            // LOG(INFO) << "      points size: " << points->size();
            float distance = 0;
            float min_distance = 1;
            float center_x=0, center_y=0, center_z=0;
            for (size_t i = 0u; i < points->rows; ++i) {
                if (!std::isfinite((points->at<float>(i,0))) ||
                    !std::isfinite((points->at<float>(i,1))) ||
                    !std::isfinite((points->at<float>(i,2)))) {
                
                    LOG(ERROR) << "Error in initalize segments. There is NaN points in segments! " ;
                    continue;
                }
                p_count += 1;
                float temp_dis = sqrt( pow(points->at<float>(i,0),2) + pow(points->at<float>(i,1),2) + pow(points->at<float>(i,2),2) );
                distance += temp_dis;
                min_distance = (min_distance>temp_dis) ? temp_dis:min_distance;
                center_x += points->at<float>(i,0);
                center_y += points->at<float>(i,1);
                center_z += points->at<float>(i,2);
            }
            // LOG(INFO) << "      mean of distance: " <<  (distance)/p_count ;
            // LOG(INFO) << "      min of distance: " <<  (min_distance);
            // LOG(INFO) << "      center: ( " <<   center_.at<float>(0,0) << ","<<   center_.at<float>(0,1) << ","<<   center_.at<float>(0,2) << ",";
            // LOG(INFO) << "      calculated center: " <<   center_x/p_count << ","<<   center_y/p_count << ","<<   center_z/p_count << ",";
            // LOG(INFO) << "      inst_confidence: " <<  inst_confidence_ ;
            // LOG(INFO) << "      obj_seg_confidence: " <<  obj_seg_confidence_;
        }
        // LOG(INFO) << "      Segments inst_confidence: " << inst_confidence <<"; obj_confidence: " << obj_seg_confidence_;
        // LOG(INFO) << "      b_box_ shape: " << b_box_.rows <<", " << b_box_.cols;
        // for (size_t i = 0u; i < b_box_.rows; ++i) {

        //     LOG(INFO) << "          B Box points: (" << b_box_.at<float>(i,0)<<", "<< b_box_.at<float>(i,1)<<", "<< b_box_.at<float>(i,2) << ")";
        // }
    }
}