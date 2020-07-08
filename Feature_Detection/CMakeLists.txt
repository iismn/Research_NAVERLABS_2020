cmake_minimum_required(VERSION 2.8)
project(Feature_Detection)

IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} INCLUDE_DIRECTORIES(DBoW3/src))
add_executable(Feature_Detection main.cpp)
target_link_libraries(Feature_Detection ${OpenCV_LIBS})
LINK_LIBRARIES(DBoW3)
