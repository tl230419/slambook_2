cmake_minimum_required(VERSION 2.8)

project(practice)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

include_directories("/usr/include/eigen3")

add_executable(prac_5 prac_5.cpp)
add_executable(prac_6 prac_6.cpp)
add_executable(prac_7 prac_7.cpp)
add_executable(prac_7_1 prac_7_1.cpp)
