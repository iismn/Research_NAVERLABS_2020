cmake_minimum_required(VERSION 2.8.3)
project(Place_Recognition)

IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

find_package(Threads)
find_package(OpenCV REQUIRED)

SET (SRCS
    dbow3_core/BowVector.cpp
    dbow3_core/Database.cpp
    dbow3_core/DescManip.cpp
    dbow3_core/FeatureVector.cpp
    dbow3_core/QueryResults.cpp
    dbow3_core/quicklz.c
    dbow3_core/ScoringObject.cpp
    dbow3_core/Vocabulary.cpp
    )


INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} dbow3_core)

# This is rather crude to link all the .o with the executables.
# I will accept a PR (pull-request) if you change this bit to
# make a static-linked library (.a) for the core files.

# Usage
add_executable( Place_Recognition main.cpp ${SRCS})
target_link_libraries( Place_Recognition  ${OpenCV_LIBS} )
