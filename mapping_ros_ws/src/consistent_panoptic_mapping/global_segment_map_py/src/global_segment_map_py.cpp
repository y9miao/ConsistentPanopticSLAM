#include "global_segment_map_py/global_segment_map_py.h"
#include <global_segment_map/common.h>
#include <opencv2/core/eigen.hpp>
#include<opencv2/opencv.hpp>
#include <Eigen/Core>

#include <chrono>  // chrono::system_clock
#include <ctime>   // localtime
#include <iomanip> // put_time
#include <fstream>


using namespace voxblox;
using namespace voxblox::voxblox_gsm;



GlobalSegmentMap_py::GlobalSegmentMap_py(std::string log_file, std::string task,
    bool use_geo_confidence,  bool use_label_confidence, 
    int inst_association, int data_association, 
    int num_threads, bool debug, int seg_graph_confidence, bool use_inst_label_connect,
    float connection_ratio_th):
    integrated_frames_count_(0u),
    task_(task),
    use_geo_confidence_(use_geo_confidence),
    use_label_confidence_(use_label_confidence),
    inst_association_(inst_association),
    data_association_(data_association),
    enable_semantic_instance_segmentation_(true),
    use_label_propagation_(true),
    log_file_(log_file),
    debug_(debug),
    seg_graph_confidence_(seg_graph_confidence),
    use_inst_label_connect_(use_inst_label_connect),
    connection_ratio_th_(connection_ratio_th)
{
    FLAGS_alsologtostderr = true;
    // GLOG_log_dir = log_file_++"/log/";
    FLAGS_log_dir = log_file_;
    // google::SetLogDestination(0,(log_file_).c_str());
    // google::SetLogDestination(2,(log_file_+"/error.log").c_str());
    // FLAGS_log_dir = log_file_.c_str();
    google::InitGoogleLogging("gms_py_test");



    if(num_threads>0 && num_threads<=tsdf_integrator_config_.integrator_threads){
        tsdf_integrator_config_.integrator_threads = num_threads;
        mesh_config_.integrator_threads = num_threads;
        visualizer_config_.thread_num = num_threads;
    }
        
    size_t integrator_threads = tsdf_integrator_config_.integrator_threads;
    LOG(INFO) << "integrator_threads: " << integrator_threads;
    map_config_.voxel_size = 0.01; // TODO yaml
    map_config_.inst_association = inst_association_;
    map_config_.voxels_per_side = 8u; // TODO yaml
    map_config_.use_inst_label_connect = use_inst_label_connect_;
    map_config_.connection_ratio_th = connection_ratio_th_;
    map_.reset(new LabelTsdfMap(map_config_));


    // Determine TSDF Label integrator parameters.
    // TSDF
    tsdf_integrator_config_.voxel_carving_enabled = false;
    tsdf_integrator_config_.allow_clear = true;
    FloatingPoint truncation_distance_factor = 5.0f; // TODO yaml
    tsdf_integrator_config_.max_ray_length_m = 3.0f; // TODO yaml
    tsdf_integrator_config_.min_ray_length_m = 0.1f; // TODO yaml
    tsdf_integrator_config_.default_truncation_distance = map_config_.voxel_size * truncation_distance_factor;
    std::string method("merged");
    tsdf_integrator_config_.enable_anti_grazing = false; // TODO yaml
    // Label
    if(use_label_confidence_)
        label_tsdf_integrator_config_.merging_min_overlap_ratio = 0.1; // TODO yaml
    else
        label_tsdf_integrator_config_.merging_min_overlap_ratio = 0.15; // TODO yaml
    label_tsdf_integrator_config_.merging_min_frame_count = 2; // TODO yaml
    label_tsdf_integrator_config_.enable_semantic_instance_segmentation = true; // TODO yaml
    // Task
    std::string class_task = task_;
    if (class_task.compare("coco80") == 0) {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kCoco80;
        BackgroundSemLabel = 0u;
    } else if (class_task.compare("nyu13") == 0) {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kNyu13;
    } else if (class_task.compare("cocoPano") == 0) {
        BackgroundSemLabel = 80;
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kCocoPano;
    } else {
        
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask::kCoco80;
    }
    integrator_.reset(new LabelTsdfConfidenceIntegrator(
      tsdf_integrator_config_, label_tsdf_integrator_config_, map_.get()
        ,use_geo_confidence_, use_label_confidence_, inst_association_, 
        data_association_, seg_graph_confidence_));

    // mesh layer and integrator settings.
    // mesh_merged_layer_.reset(new MeshLayer(map_->block_size()));
    mesh_label_layer_.reset(new MeshLayer(map_->block_size()));
    mesh_semantic_layer_.reset(new MeshLayer(map_->block_size()));
    mesh_instance_layer_.reset(new MeshLayer(map_->block_size()));
    mesh_confidence_layer_.reset(new MeshLayer(map_->block_size()));

    // label_tsdf_mesh_config_.color_scheme =
    //   MeshLabelIntegrator::ColorScheme::kMerged;
    // mesh_merged_integrator_.reset(
    //   new MeshLabelIntegrator(mesh_config_, label_tsdf_mesh_config_, map_.get(),
    //                           mesh_merged_layer_.get(), &need_full_remesh_));
    label_tsdf_mesh_config_.color_scheme =
        MeshLabelIntegrator::ColorScheme::kLabel;
    mesh_label_integrator_.reset(new MeshLabelIntegrator(
        mesh_config_, label_tsdf_mesh_config_, map_.get(),
        mesh_label_layer_.get(), &need_full_remesh_));
    label_tsdf_mesh_config_.color_scheme =
        MeshLabelIntegrator::ColorScheme::kSemantic;
    mesh_semantic_integrator_.reset(new MeshLabelIntegrator(
        mesh_config_, label_tsdf_mesh_config_, map_.get(),
        mesh_semantic_layer_.get(), &need_full_remesh_));
    label_tsdf_mesh_config_.color_scheme =
        MeshLabelIntegrator::ColorScheme::kInstance;
    mesh_instance_integrator_.reset(new MeshLabelIntegrator(
        mesh_config_, label_tsdf_mesh_config_, map_.get(),
        mesh_instance_layer_.get(), &need_full_remesh_));

    label_tsdf_mesh_config_.color_scheme =
      MeshLabelIntegrator::ColorScheme::kLabelConfidence;
    mesh_confidence_integrator_.reset(
      new MeshLabelIntegrator(mesh_config_, label_tsdf_mesh_config_, map_.get(),
                              mesh_confidence_layer_.get(), &need_full_remesh_));

    // meshes
    std::vector<std::shared_ptr<MeshLayer>> mesh_layers;
    mesh_layers.push_back(mesh_instance_layer_);
    mesh_layers.push_back(mesh_label_layer_);
    mesh_layers.push_back(mesh_semantic_layer_);
    // mesh_layers.push_back(mesh_confidence_layer_);
    // mesh_layers.push_back(mesh_merged_layer_);
    mesh_layer_updated_ = false;
    LOG(INFO) << "  BackgroundSemLabel: " << int(BackgroundSemLabel);
    LOG(INFO) << "  use geometric confidence: " << use_geo_confidence_;
    LOG(INFO) << "  use label confidence: " << use_label_confidence_;
    LOG(INFO) << "  inst association: " << inst_association_;
    LOG(INFO) << "  inst association: " << 
        integrator_->semantic_instance_label_fusion_ptr_->inst_association_;
    LOG(INFO) << "  data association: " << data_association_;
    LOG(INFO) << "  data association: " << integrator_->data_association_;
    LOG(INFO) << "  seg_graph_confidence: " << seg_graph_confidence_;
    LOG(INFO) << "  seg_graph_confidence: " << integrator_->seg_graph_confidence_;
    if( inst_association_ == 3 || inst_association_ == 4)
    {
      integrator_->semantic_instance_label_fusion_ptr_->initSegGraph();
      LOG(INFO) << "  use_inst_label_connect: " << use_inst_label_connect_;
      LOG(INFO) << "  use_inst_label_connect: " << 
          integrator_->semantic_instance_label_fusion_ptr_->use_inst_label_connect_;
      LOG(INFO) << "  connection_ratio_th_: " << connection_ratio_th_;
      LOG(INFO) << "  connection_ratio_th_: " << 
          integrator_->semantic_instance_label_fusion_ptr_->connection_ratio_th_;
    }
    // visualizer 
    std::vector<double> camera_position = {
        -1.41162,    6.28602,   -0.300336,
        -1.49346,    0.117437,   0.0843885,
        0.0165199, -0.0624571, -0.997911}; // TODO yaml
    std::vector<double> clip_distances = {0.1, 8.86051}; // TODO yaml
    // std::vector<double> clip_distances = {1.79126, 8.86051}; // TODO yaml
    double update_mesh_every_n_sec = 0.0;

    bool save_visualizer_frames = false;
    visualizer_mesh_ = std::make_shared< Visualizer >(mesh_layers, &mesh_layer_updated_, &mesh_layer_mutex_,
        camera_position, clip_distances, save_visualizer_frames);
    visualizer_pcl_ = std::make_shared< PCLSemVisualizer >(
        visualizer_config_, map_.get(), camera_position, clip_distances);
    if(debug_)
    {
        viz_mesh_thread_ = std::thread(&Visualizer::visualizeMesh, visualizer_mesh_);
        // viz_pcl_thread_ = std::thread(&PCLSemVisualizer::visualizePointClouds, visualizer_pcl_);
    }
    LOG(INFO) << "  Memory usage at init: " << getValue() << " kB";
}

void GlobalSegmentMap_py::insertSegments(
        pybind11::array& points, // float
        // pybind11::array& colors, // rgba uint8_t
        // pybind11::array& geometry_confidence, //float
        pybind11::array& b_box, //float
        InstanceLabel instance_label, //uint16_t
        SemanticLabel semantic_label, //uint8_t
        ObjSegConfidence inst_confidence,
        ObjSegConfidence obj_seg_confidence,
        pybind11::array &T_G_C,
        bool is_thing){
    // LOG(INFO) << "  Memory usage before insertSegments: " << getValue() << " kB";
    cv::Mat T_G_C_mat = cvnp::nparray_to_mat(T_G_C);
    Eigen::Matrix<float, 4, 4> T_G_C_eigen;
    cv::cv2eigen(T_G_C_mat, T_G_C_eigen);
    Transformation T_G_C_voxblox(T_G_C_eigen);

    cv::Mat points_mat = cvnp::nparray_to_mat(points);
    // cv::Mat colors_mat = cvnp::nparray_to_mat(colors);
    // cv::Mat geometry_confidence_mat = cvnp::nparray_to_mat(geometry_confidence);
    cv::Mat b_box_mat = cvnp::nparray_to_mat(b_box);
    Segment* segment = nullptr;
    segment = new SegmentConfidence(&points_mat, &b_box_mat, 
      instance_label, semantic_label, T_G_C_voxblox, inst_confidence, obj_seg_confidence, is_thing);
    // if(instance_label!=0)
    //   LOG(INFO) <<  "   New segment sem " << int(segment->semantic_label_) << " inst " 
    //   << int(segment->instance_label_);
    // CHECK_NOTNULL(segment);
    segments_to_integrate_.push_back(segment);
    // LOG(INFO) << "  Memory usage after insertSegments: " << getValue() << " kB";
}

bool GlobalSegmentMap_py::integrateFrame()
{
  bool whether_merge_alias = false;
  LOG(INFO) << "Integrating frame n." << ++integrated_frames_count_ ;
  LOG(INFO) << "  Memory usage before integrateFrame: " << getValue() << " kB";

  auto time_start = std::chrono::system_clock::now();
  for (Segment* segment : segments_to_integrate_) {
    
    integrator_->computeSegmentLabelCandidatesConfidence(
        segment, &segment_label_candidates, &segment_merge_candidates_);
  }
  auto time_end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration<double>(time_end-time_start).count();
  LOG(INFO) << "  computeSegmentLabelCandidatesConfidence cost: " << duration << " seconds";
  // LOG(INFO) << "  Confidence candidate. ";
  // for (auto label_it = segment_label_candidates.begin(); 
  //         label_it != segment_label_candidates.end();++label_it) {
  //         for (auto segment_it = label_it->second.begin();
  //             segment_it != label_it->second.end(); segment_it++) {
  //         LOG(INFO) << "    Label " << label_it->first << " - seg.size " 
  //           << (segment_it->first)->points_C_.size() << " - confi " << segment_it->second;
  //     }
  // }

  // true in default
  
  if (use_label_propagation_) {
    time_start = std::chrono::system_clock::now();
    integrator_->decideLabelPointCloudsConfidence(&segments_to_integrate_,
                                        &segment_label_candidates,
                                        &segment_merge_candidates_);

    time_end = std::chrono::system_clock::now();
    duration = std::chrono::duration<double>(time_end-time_start).count();
    LOG(INFO) << "  decideLabelPointCloudsConfidence cost: " << duration << " seconds";
    // LOG_EVERY_N(INFO, 1) << "  Decided labels for " << segments_to_integrate_.size()
    //     << std::fixed << std::setprecision(4)
    //     << " pointclouds in " << duration << " seconds.";
  }
  constexpr bool kIsFreespacePointcloud = false;
  Transformation T_G_C = segments_to_integrate_.at(0)->T_G_C_;

  Transformation T_Gicp_C = T_G_C;

  {
    size_t seg_count = 0;
    auto time_start = std::chrono::system_clock::now();

    std::lock_guard<std::mutex> label_tsdf_layers_lock(
        label_tsdf_layers_mutex_);


    for (Segment* segment : segments_to_integrate_) {
      CHECK_NOTNULL(segment);
      segment->T_G_C_ = T_Gicp_C;
      integrator_->integratePointCloudConfidence(
        segment->T_G_C_, segment->points_C_,
        dynamic_cast<SegmentConfidence*>(segment)->geometry_confidence_,
        dynamic_cast<SegmentConfidence*>(segment)->seg_label_confidence_,
        segment->colors_, segment->label_,
        kIsFreespacePointcloud); // TODO for confidence  

      // LOG_EVERY_N(INFO, 100) << "    segments n." << seg_count++ << " with " 
      //   << segment->points_C_.size() << " points"; 
      // if(segment->instance_label_!=0)
      // {
      //   LOG_EVERY_N(INFO, 1) << " segment segseg confidence: " << 
      //     dynamic_cast<SegmentConfidence*>(segment)->seg_label_confidence_;
      //   LOG_EVERY_N(INFO, 1) << " segment obj_seg_confidence confidence: " << 
      //     dynamic_cast<SegmentConfidence*>(segment)->obj_seg_confidence_;
      // }

    }

      auto time_end = std::chrono::system_clock::now();
      auto duration = std::chrono::duration<double>(time_end-time_start).count();
      LOG(INFO) << "  integratePointCloudConfidence cost: " << duration << " seconds";

    // LOG_EVERY_N(INFO, 1) << "  Integrated " << segments_to_integrate_.size()
    //     << " pointclouds in " << duration << " secs. ";

    // LOG_EVERY_N(INFO, 1) << "  The map contains "
    //     << map_->getTsdfLayerPtr()->getNumberOfAllocatedBlocks()
    //     << " tsdf and "
    //     << map_->getLabelLayerPtr()->getNumberOfAllocatedBlocks()
    //     << " label blocks.";
    
    // LOG(INFO) << "  Check for label merge";
    if(data_association_!=0)
      {whether_merge_alias = integrator_->mergeLabelConfidence(&merges_to_publish_);}
    else
      {integrator_->mergeLabels(&merges_to_publish_);}

    integrator_->getLabelsToPublish(&segment_labels_to_publish_);

  }
    // LOG(INFO) << " PairWiseConfidence: ";

    // for( LLMapIt label_it = integrator_-> pairwise_confidence_.begin();
    //       label_it!=integrator_-> pairwise_confidence_.end(); label_it++)
    // {
    //   for(LMapIt pair_it=label_it->second.begin(); pair_it!=label_it->second.end(); pair_it++)
    //   {
    //       LOG(INFO) << "    Label " <<  int(label_it->first) << "- "<< int(pair_it->first) << 
    //           ": " << pair_it->second; 
    //   }
      
    // }

    // log memory usage
    // LOG(INFO) << "  Memory usage: " << getValue() << " kB";
    LOG(INFO) << "  Memory usage after integrateFrame: " << getValue() << " kB";
  return whether_merge_alias;
}

void GlobalSegmentMap_py::LogSegmentsLabels()
{
  LOG(INFO) << " LogSegmentsInformation: "; 
    for (Segment* segment : segments_to_integrate_) {
      Label global_label = segment->label_;
      SemanticLabel seman_label = integrator_->semantic_instance_label_fusion_ptr_->getSemanticLabel(global_label) ;
          LOG(INFO) << " Label"<<int(global_label) << " ; size: " << segment->points_C_.size() << " ; semantic label: " << int(seman_label);
    }

}

void GlobalSegmentMap_py::clearTemporaryMemory()
{
    segment_merge_candidates_.clear();
    segment_label_candidates.clear();
    for (Segment* segment : segments_to_integrate_) {
      delete segment;
    }
    segments_to_integrate_.clear();
    merges_to_publish_.clear();
    segment_labels_to_publish_.clear();
    LOG(INFO) << "  Memory usage after clearTemporaryMemory: " << getValue() << " kB";
}

bool GlobalSegmentMap_py::generateMesh(std::string mesh_file_folder,std::string frame_num)
{
  bool clear_mesh = true; // default

  std::lock_guard<std::mutex> mesh_layer_lock(mesh_layer_mutex_);
  
  std::lock_guard<std::mutex> label_tsdf_layers_lock(
      label_tsdf_layers_mutex_);

  bool only_mesh_updated_blocks = false;
  constexpr bool clear_updated_flag = true;
  // mesh_merged_integrator_->generateMesh(only_mesh_updated_blocks,
  //                                       clear_updated_flag);

  mesh_label_integrator_->generateMesh(only_mesh_updated_blocks,
                                        clear_updated_flag);
  mesh_semantic_integrator_->generateMesh(only_mesh_updated_blocks,
                                          clear_updated_flag);
  mesh_instance_integrator_->generateMesh(only_mesh_updated_blocks,
                                          clear_updated_flag);
  // mesh_confidence_integrator_->generateMesh(only_mesh_updated_blocks,
  //                                     clear_updated_flag);
  mesh_layer_updated_ = true;
  
  // bool success = outputMeshLayerAsPly(mesh_file_folder + "/confidence_mesh_"+frame_num+".ply", false,
  //                                     *mesh_confidence_layer_);
  bool success = outputMeshLayerAsPly(mesh_file_folder + "/label_mesh_"+frame_num+".ply", false,
                                  *mesh_label_layer_);
  success &= outputMeshLayerAsPly(mesh_file_folder + "/semantic_mesh_"+frame_num+".ply", false,
                                  *mesh_semantic_layer_);
  success &= outputMeshLayerAsPly(mesh_file_folder + "/instance_mesh_"+frame_num+".ply", false,
                                  *mesh_instance_layer_);
  if (success) {
    LOG(INFO) << "Output file as PLY: " << mesh_file_folder.c_str();
  } else {
    LOG(INFO) << "Failed to output mesh as PLY: " << mesh_file_folder.c_str();
  }
    
  return success;
}

void GlobalSegmentMap_py::updateVisualization()
{
  std::lock_guard<std::mutex> mesh_layer_lock(mesh_layer_mutex_);
  {
    std::lock_guard<std::mutex> label_tsdf_layers_lock(
        label_tsdf_layers_mutex_);

    bool need_full_remesh_ = true;
    bool only_mesh_updated_blocks = true;
    if (need_full_remesh_) {
      only_mesh_updated_blocks = false;
      need_full_remesh_ = false;
    }

    bool clear_updated_flag = false;
    mesh_layer_updated_ |= mesh_label_integrator_->generateMesh(
        only_mesh_updated_blocks, clear_updated_flag);
    mesh_layer_updated_ |= mesh_instance_integrator_->generateMesh(
        only_mesh_updated_blocks, clear_updated_flag);
    mesh_layer_updated_ |= mesh_semantic_integrator_->generateMesh(
        only_mesh_updated_blocks, clear_updated_flag);
    clear_updated_flag = true;
    // TODO(ntonci): Why not calling generateMesh instead?
    // mesh_layer_updated_ |= mesh_merged_integrator_->generateMesh(
    //     only_mesh_updated_blocks, clear_updated_flag);

    // mesh_layer_updated_ |= mesh_confidence_integrator_->generateMesh(
    //     only_mesh_updated_blocks, clear_updated_flag);

  }
}
void GlobalSegmentMap_py::updateVisualizationPCL()
{
  visualizer_pcl_->updatePointClouds();
}

void GlobalSegmentMap_py::LogLabelInformation()
{
  integrator_->cleanStaleLabels();
  SemanticInstanceLabelFusion* log_semantic_instance_label_fusion_ptr_ = 
    integrator_->semantic_instance_label_fusion_ptr_;
  LOG(ERROR) << "Log Label Information: "; 

  if(inst_association_ == 3 || inst_association_ == 4)
  {
    for(auto label_it = log_semantic_instance_label_fusion_ptr_->label_frames_count_.begin();
        label_it!=log_semantic_instance_label_fusion_ptr_->label_frames_count_.end(); label_it++)
    {
      InstanceLabel instance_label = log_semantic_instance_label_fusion_ptr_->getInstanceLabel(label_it->first, 0.1f);
      if(instance_label != BackgroundLabel)
      {
        SemanticLabel semantic_label = log_semantic_instance_label_fusion_ptr_->getSemanticLabel(label_it->first);
        LOG(ERROR) << " Label: " << int(label_it->first) <<" Sem: " << int(semantic_label)<<" Inst: " << std::setfill('0') << std::setw(5) << int(instance_label); 
      }
    }

    log_semantic_instance_label_fusion_ptr_->logSegGraphInfo(log_file_);
    LogLabelInitialGuess(log_file_);
  }
  else
  {
    for(auto label_it = log_semantic_instance_label_fusion_ptr_->label_frames_count_.begin();
        label_it!=log_semantic_instance_label_fusion_ptr_->label_frames_count_.end(); label_it++)
    {
      InstanceLabel instance_label = log_semantic_instance_label_fusion_ptr_->getInstanceLabel(label_it->first, 0.1f);
      if(instance_label != BackgroundLabel)
      {
        SemanticLabel semantic_label = log_semantic_instance_label_fusion_ptr_->getSemanticLabel(label_it->first);
        LOG(ERROR) << " Label: " << int(label_it->first) <<" Sem: " << int(semantic_label)<<" Inst: " << std::setfill('0') << std::setw(5) << int(instance_label); 
        log_semantic_instance_label_fusion_ptr_->logLabelSemanticInstanceCountInfo(label_it->first);
      }
    }
  }


}

void GlobalSegmentMap_py::LogMeshColors()
{
  std::set<InstanceLabel> instance_label_set;

  const SemanticInstanceLabelFusion* log_semantic_instance_label_fusion_ptr_ = 
    integrator_->semantic_instance_label_fusion_ptr_;
  LOG(ERROR) << "LogLabelColorInformation: "; 
  Color label_color; 
  mesh_label_integrator_->label_color_map_.getColor(1, &label_color);
  LOG(ERROR) << " Label: " << 1 <<" Color: (" 
    << int(label_color.r)<<","<<int(label_color.g)<<","<<int(label_color.b)<<")";
  for(auto label_it = log_semantic_instance_label_fusion_ptr_->label_frames_count_.begin();
      label_it!=log_semantic_instance_label_fusion_ptr_->label_frames_count_.end(); label_it++)
  {
    InstanceLabel instance_label = log_semantic_instance_label_fusion_ptr_->getInstanceLabel(label_it->first, 0.1f);
    if(instance_label != BackgroundLabel)
    {
      if(instance_label_set.find(instance_label) == instance_label_set.end())
        instance_label_set.insert(instance_label);

      Color label_color; 
      mesh_label_integrator_->label_color_map_.getColor(label_it->first, &label_color);
      LOG(ERROR) << " Label: " << int(label_it->first) <<" Color: (" 
        << int(label_color.r)<<","<<int(label_color.g)<<","<<int(label_color.b)<<")";
    }
  }

  LOG(ERROR) << "LogInstanceColorInformation: "; 
  for(auto inst_it=instance_label_set.begin(); inst_it!=instance_label_set.end(); inst_it++)
  {
      Color inst_color; 
      mesh_instance_integrator_->instance_color_map_.getColor(*inst_it, &inst_color);
      LOG(ERROR) << " Instance: " << int(*inst_it) <<" Color: (" 
        << int(inst_color.r)<<","<<int(inst_color.g)<<","<<int(inst_color.b)<<")";
  }

  if(inst_association_ == 3)
  {
    // log sem connection map 
  }
}

void GlobalSegmentMap_py::LogLabelInitialGuess(std::string log_path)
{
  std::string label_inital_log = log_path + "/LabelInitialGuess.txt";
  LOG(ERROR) << "Label initial guess into LabelInitialGuess.txt: "; 
  std::ofstream log_file_io;
  log_file_io.open(label_inital_log.c_str());
  if ( log_file_io.is_open() )
  {
    log_file_io << "# label initial instance guess " << std::endl;
    log_file_io << "# format: label semantic_label instance_label r g b " << std::endl;

    SemanticInstanceLabelFusion* log_semantic_instance_label_fusion_ptr_ = 
      integrator_->semantic_instance_label_fusion_ptr_;
    for(auto label_it = log_semantic_instance_label_fusion_ptr_->label_frames_count_.begin();
        label_it!=log_semantic_instance_label_fusion_ptr_->label_frames_count_.end(); label_it++)
    {
      InstanceLabel instance_label = log_semantic_instance_label_fusion_ptr_->getInstanceLabel(label_it->first, 0.1f);
      SemanticLabel semantic_label = log_semantic_instance_label_fusion_ptr_->getSemanticLabel(label_it->first);
      Color label_color; 
      mesh_label_integrator_->label_color_map_.getColor(label_it->first, &label_color);
      log_file_io << std::setfill('0') << std::setw(5) << int(label_it->first) << " "
        << int(semantic_label) << " "
        << std::setfill('0') << std::setw(5) << int(instance_label) << " "
        << int(label_color.r)<<" "<<int(label_color.g)<<" "<<int(label_color.b) << std::endl;;
      
    }
  }
}

void GlobalSegmentMap_py::outputLog(std::string log_info)
{
  LOG(INFO) << log_info;
}

PYBIND11_MODULE(consistent_gsm, m) {
    m.doc() = "pybind11 for consistent global segmentation map"; // optional module docstring
    
    pybind11::class_<GlobalSegmentMap_py>(m, "GlobalSegmentMap_py")
        .def(pybind11::init<std::string, std::string, bool, bool, int, int, int, bool, int, bool, float>())
        .def("insertSegments", &GlobalSegmentMap_py::insertSegments)
        .def("integrateFrame", &GlobalSegmentMap_py::integrateFrame)
        .def("LogSegmentsLabels", &GlobalSegmentMap_py::LogSegmentsLabels)
        .def("clearTemporaryMemory", &GlobalSegmentMap_py::clearTemporaryMemory)
        .def("generateMesh", &GlobalSegmentMap_py::generateMesh)
        .def("outputLog", &GlobalSegmentMap_py::outputLog)
        .def("updateVisualization", &GlobalSegmentMap_py::updateVisualization)
        .def("updateVisualizationPCL", &GlobalSegmentMap_py::updateVisualizationPCL)
        .def("LogLabelInformation", &GlobalSegmentMap_py::LogLabelInformation)
        .def("LogMeshColors", &GlobalSegmentMap_py::LogMeshColors);
}