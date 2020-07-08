cmake_minimum_required(VERSION 2.8.3)
project(Place_Recognition_FBoW)

IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

find_package(Threads)
find_package(OpenCV REQUIRED)

SET (SRCS
    FBoW_core/fbow.cpp
    FBoW_core/vocabulary_creator.cpp
    )


INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} FBoW_core)

# This is rather crude to link all the .o with the executables.
# I will accept a PR (pull-request) if you change this bit to
# make a static-linked library (.a) for the core files.

# Usage
add_executable( Place_Recognition_FBoW main.cpp ${SRCS})
target_link_libraries( Place_Recognition_FBoW  ${OpenCV_LIBS} )
