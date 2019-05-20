cmake_minimum_required(VERSION 2.8)

project(hello)

add_subdirectory(src)
add_subdirectory(libhello)

set(CMAKE_BUILD_TYPE "Release")
