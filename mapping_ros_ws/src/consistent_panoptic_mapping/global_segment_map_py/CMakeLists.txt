cmake_minimum_required(VERSION 2.8.3)
project(consistent_gsm)

add_definitions(-std=c++14)

find_package(catkin_simple REQUIRED)
catkin_simple()
find_package(catkin REQUIRED COMPONENTS
global_segment_map
gsm_node
opencv3_catkin
# pybind11_catkin
)

## Declare the catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES
  ${PROJECT_NAME}
  CATKIN_DEPENDS
    global_segment_map
    gsm_node
    opencv3_catkin
    voxblox
    voxblox_ros
    # pybind11_catkin
)

##########
# cvnp   #
##########
add_subdirectory(cvnp)

##########
# Pybind #
##########
include_directories(cvnp)
find_package(pybind11 REQUIRED)

##########
# this project   #
##########
include_directories(include)
cs_add_library(${PROJECT_NAME}_LIB
  # src/segment_graph.cpp //moved to global_segment_map
  src/segment_confidence.cpp
  src/label_tsdf_confidence_integrator.cpp
  src/utils/pcl_semantic_visualizers.cpp
)

pybind11_add_module(${PROJECT_NAME} src/global_segment_map_py.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE cvnp ${catkin_LIBRARIES} ${PROJECT_NAME}_LIB)
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/${PROJECT_NAME}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}/${PROJECT_NAME}
)
# install(
#   TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/${PROJECT_NAME}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/${PROJECT_NAME}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}/${PROJECT_NAME}
# )

set(ENV{PYTHONPATH} ENV{PYTHONPATH} "${CATKIN_PACKAGE_LIB_DESTINATION}")
# list(APPEND PYTHONPATH "${CATKIN_PACKAGE_LIB_DESTINATION}") 
MESSAGE(STATUS "PYTHONPATH: ${PYTHONPATH}")
MESSAGE(STATUS "CATKIN_PACKAGE_LIB_DESTINATION: ${CATKIN_PACKAGE_LIB_DESTINATION}")
# catkin_python_setup(PROGRAMS ${PROJECT_NAME})