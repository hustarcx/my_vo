cmake_minimum_required(VERSION 2.8)
project(my_vo)

set( CMAKE_BUILD_TYPE "Release")
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(OpenCV 3.1 REQUIRED)
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

add_executable(vo_feature_extation vo_feature_extation.cpp)
target_link_libraries( vo_feature_extation ${OpenCV_LIBS} )

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )