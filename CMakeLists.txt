cmake_minimum_required( VERSION 2.8 )
project(myslam)

# colors, because why not :)
string(ASCII 27 Esc)
set(RED "${Esc}[31m")
set(GREEN "${Esc}[32m")
set(ENDL "${Esc}[m")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
MESSAGE("${GREEN}Build type: ${CMAKE_BUILD_TYPE}${ENDL}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++11 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################
#Eigen
include_directories("/usr/include/eigen3")

#Boost
find_package(Boost COMPONENTS filesystem REQUIRED)
include_directories(${Boost_INLCLUDE_DIRS})

#OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_Include_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# glog: 实现应用级别的logging
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# caffe (default path: ~/caffe)
add_definitions( -DCPU_ONLY=1 )
if (NOT Caffe_ROOT_DIR)
	set(Caffe_ROOT_DIR $ENV{HOME}/caffe)
endif()
MESSAGE("${GREEN}Using Caffe_ROOT_DIR: ${Caffe_ROOT_DIR}${ENDL}")
set(Caffe_LIBS ${Caffe_ROOT_DIR}/build/lib/libcaffe.so)
set(Caffe_INCLUDE_DIRS ${Caffe_ROOT_DIR}/build/install/include)
include_directories(${Caffe_INCLUDE_DIRS})

# cuda (optional)
find_package(CUDA QUIET)
if (${CUDA_FOUND})
 	include_directories(${CUDA_INCLUDE_DIRS})
	MESSAGE("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
endif()


###################    third party libs    #####################
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
        ${Caffe_LIBS}
	${Boost_LIBRARIES}
        )

if (${CUDA_FOUND})
target_link_libraries(THIRD_PARTY_LIBS
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
)
endif()


#################### Default model download ####################
message( "\n\n${GREEN}Downloading default model if not already downloaded\n\n${ENDL}" )
execute_process(
	COMMAND ./get_model.sh
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)



enable_testing()
############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
# add_subdirectory(test)
add_subdirectory(app)
