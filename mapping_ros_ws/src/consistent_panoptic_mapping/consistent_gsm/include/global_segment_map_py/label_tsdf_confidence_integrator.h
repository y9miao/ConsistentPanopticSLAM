#ifndef LABEL_TSDF_CONFIDENCE_INTEGRATOR_H_
#define LABEL_TSDF_CONFIDENCE_INTEGRATOR_H_

#include <global_segment_map/label_tsdf_integrator.h>
#include <global_segment_map_py/segment_confidence.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
namespace voxblox {

class LabelTsdfConfidenceIntegrator : public LabelTsdfIntegrator{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LabelTsdfConfidenceIntegrator(const Config& tsdf_config,
                    const LabelTsdfConfig& label_tsdf_config,
                    LabelTsdfMap* map,
                    bool use_geo_confidence,
                    bool use_label_confidence,
                    int inst_association,
                    int data_association,
                    int seg_graph_confidence)
    : LabelTsdfIntegrator(tsdf_config,label_tsdf_config,map),
    use_geo_confidence_(use_geo_confidence),
    use_label_confidence_(use_label_confidence),
    inst_association_(inst_association),
    data_association_(data_association),
    seg_graph_confidence_(seg_graph_confidence){}

    void integratePointCloudConfidence(
        const Transformation& T_G_C,
        const Pointcloud& points_C,
        const std::vector<GeometricConfidence>& geometric_confidence,
        const SegSegConfidence& label_confidence,
        const Colors& colors,
        const Label& label,
        const bool freespace_points);

    void integrateRaysConfidence(
        const Transformation& T_G_C, 
        const Pointcloud& points_C,
        const std::vector<GeometricConfidence>& geometric_confidence,
        const SegSegConfidence& label_confidence,
        const Colors& colors, 
        const Label& label, 
        const bool enable_anti_grazing,
        const bool clearing_ray, const VoxelMap& voxel_map,
        const VoxelMap& clear_map);

    void integrateVoxelsConfidence(
        const Transformation& T_G_C, 
        const Pointcloud& points_C,
        const std::vector<GeometricConfidence>& geometric_confidence,
        const SegSegConfidence& label_confidence,
        const Colors& colors, 
        const Label& label, 
        const bool enable_anti_grazing,
        const bool clearing_ray, 
        const VoxelMap& voxel_map,
        const VoxelMap& clear_map, 
        const size_t thread_idx);

    void integrateVoxelConfidence(
        const Transformation& T_G_C, 
        const Pointcloud& points_C,
        const std::vector<GeometricConfidence>& geometric_confidence,
        const SegSegConfidence& label_confidence,
        const Colors& colors, 
        const Label& label, 
        const bool enable_anti_grazing,
        const bool clearing_ray,
        const VoxelMapElement& global_voxel_idx_to_point_indices,
        const VoxelMap& voxel_map);
        
    void updateLabelVoxelConfidence(
        const Point& point_G,
        const Label& label,
        LabelVoxel* label_voxel,
        const LabelConfidence& confidence);
    void updateLabelVoxelConfidence(
        const Point& point_G,
        const FloatingPoint& point_weight,
        const Label& label,
        LabelVoxel* label_voxel,
        const LabelConfidence& confidence);

    void addVoxelLabelConfidenceSmart(
        const Label& label, 
        const LabelConfidence& confidence,
        LabelVoxel* label_voxel) ;

    void computeSegmentLabelCandidatesConfidence(
        Segment* segment, 
        std::map<Label, std::map<Segment*, SegSegConfidence>>* candidates_confidence,
        std::map<Segment*, std::vector<Label>>* segment_merge_candidates,
        const std::set<Label>& assigned_labels = std::set<Label>());

    void increaseLabelCountForSegment(
        Segment* segment, 
        const Label& label, 
        const int segment_points_count,
        std::map<Label, std::map<Segment*, SegSegConfidence>>* candidates_confidence,
        std::unordered_set<Label>* merge_candidate_labels);

    void increaseLabelConfidenceForSegment(
        Segment* segment, 
        const Label& label, 
        const int& segment_points_size,
        const SegSegConfidence confidence, 
        std::map<Label, std::map<Segment*, SegSegConfidence>>* candidates_confidence,
        std::unordered_set<Label>* merge_candidate_labels);

    void decideLabelPointCloudsConfidence(
        std::vector<Segment*>* segments_to_integrate,
        std::map<Label, std::map<Segment*, SegSegConfidence>>* candidates_confidence,
        std::map<Segment*, std::vector<Label>>* segment_merge_candidates);

    bool getNextSegmentLabelPairWithConfidence(
        std::set<Segment*, SegmentConfidence::PtrCompare>& labelled_segments,
        std::set<Label>* assigned_labels,
        std::map<Label, std::map<Segment*, SegSegConfidence>>* candidates_confidence,
        std::map<Segment*, std::vector<Label>>* segment_merge_candidates,
        std::pair<Segment*, Label>* segment_label_pair,
        SegSegConfidence* pair_confidence);

    void increasePairwiseConfidenceCountSemantics( const std::vector<Label>& merge_candidates ); 


    void IncreaseLabelInstanceMapCount(
        std::set<Segment*, SegmentConfidence::PtrCompare>* labelled_segments);
    void IncreaseSegGraphConfidence(
        std::set<Segment*, SegmentConfidence::PtrCompare>* labelled_segments);
    void ContructSegGraphInstConfidenceMap(
        std::set<Segment*, SegmentConfidence::PtrCompare>* inst_segments, 
        LLConfidenceMap* label_label_confidence_map, bool is_thing);
    void IncreaseLabelInstanceMapConfidence2(
        std::set<Segment*, SegmentConfidence::PtrCompare>* labelled_segments);
    void updateLabelClassInstanceConfidence(
        std::set<Segment*, SegmentConfidence::PtrCompare>* labelled_segments);
    void updateLabelInstanceClassConfidence(
        std::set<Segment*, SegmentConfidence::PtrCompare>* labelled_segments);

    bool mergeLabelConfidence(LLSet* merges_to_publish);
    bool getNextMergeConfidence(Label* new_label, Label* old_label);
    void updatePairwiseConfidenceAfterMerging(const Label& new_label, const Label& old_label);

    void cleanStaleLabels();

    bool use_geo_confidence_ = false;
    bool use_label_confidence_ = false;
    int inst_association_ = 0;
    int data_association_ = 0;
    int seg_graph_confidence_ = 0;
};


}

#endif  // LABEL_TSDF_CONFIDENCE_INTEGRATOR_H_


