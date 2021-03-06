cmake_minimum_required(VERSION 2.8)

project(vo1)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.1 REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	${G2O_INCLUDE_DIRS}
	${CSPARSE_INCLUDE_DIR}
	"/usr/include/eigen3/"
	${Sophus_INCLUDE_DIRS}
	${CERES_INCLUDE_DIRS}
)

add_executable(feature_extraction feature_extraction.cpp)
target_link_libraries(feature_extraction ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d 
	${OpenCV_LIBS}
	${CSPARSE_LIBRARY}
	g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d 
	${OpenCV_LIBS}
	${CSPARSE_LIBRARY}
	g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)

add_executable(prac_7_2 prac_7_2.cpp)
target_link_libraries(prac_7_2 ${OpenCV_LIBS})

# error, unsolved
#add_executable(prac_7_3 prac_7_3.cpp)
#target_link_libraries(prac_7_3 ${OpenCV_LIBS})

add_executable(prac_7_6 prac_7_6.cpp)
target_link_libraries(prac_7_6 
	${OpenCV_LIBS}
	${CSPARSE_LIBRARY}
	g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)

add_executable(prac_7_7 prac_7_7.cpp)
target_link_libraries(prac_7_7 
	${OpenCV_LIBS}
	${CSPARSE_LIBRARY}
	g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)

# error, unsolved
#add_executable(prac_7_9_pnp prac_7_9_pnp.cpp)
#target_link_libraries(prac_7_9_pnp 
#	${OpenCV_LIBS}
#	${CSPARSE_LIBRARY}
#	g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
#	${Sophus_LIBRARIES}
#)
#add_executable(prac_7_9_icp prac_7_9_icp.cpp)
#target_link_libraries(prac_7_9_icp 
#	${OpenCV_LIBS}
#	${CSPARSE_LIBRARY}
#	g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
#	${Sophus_LIBRARIES}
#)

add_executable(prac_7_10_pnp prac_7_10_pnp.cpp)
target_link_libraries(prac_7_10_pnp 
	${OpenCV_LIBS}
	${CSPARSE_LIBRARY}
	g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
	${CERES_LIBRARIES}
)
install(TARGETS prac_7_10_pnp RUNTIME DESTINATION bin)

add_executable(prac_7_10_icp prac_7_10_icp.cpp)
target_link_libraries(prac_7_10_icp 
	${OpenCV_LIBS}
	${CSPARSE_LIBRARY}
	g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
	${CERES_LIBRARIES}
)
install(TARGETS prac_7_10_icp RUNTIME DESTINATION bin)
