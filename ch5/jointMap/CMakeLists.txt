cmake_minimum_required(VERSION 2.8)

project(jointMap)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

find_package(PCL REQUIRED COMPONENT common io)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(jointMap jointMap.cpp)
target_link_libraries(jointMap ${OpenCV_LIBS} ${PCL_LIBRARIES})
