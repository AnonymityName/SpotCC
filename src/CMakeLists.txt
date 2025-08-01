cmake_minimum_required(VERSION 3.17)

project(cc-clients LANGUAGES C CXX)

# Use C++17 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# Options
#
option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)
option(TRITON_USE_THIRD_PARTY "Use local version of third party libraries" ON)
option(TRITON_KEEP_TYPEINFO "Keep typeinfo symbols by disabling ldscript" OFF)
option(TRITON_ENABLE_ZLIB "Include ZLIB library in build" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/9406a60c7839052e4944ea4dbc8344762a89f9bd.zip
)

set(TRITON_ENABLE_CC_GRPC ON)
if(TRITON_ENABLE_CC_GRPC)
  set(TRITON_COMMON_ENABLE_PROTOBUF ON)
  set(TRITON_COMMON_ENABLE_GRPC ON)

endif() # TRITON_ENABLE_CC_GRPC

FetchContent_MakeAvailable(repo-common)


#
# CUDA
#
# if(TRITON_ENABLE_GPU)
#   find_package(CUDAToolkit REQUIRED)
# endif() # TRITON_ENABLE_GPU

#
# libcurl
#
  find_package(CURL REQUIRED)
  message(STATUS "Using curl ${CURL_VERSION}")

#
# Protobuf
#
if(TRITON_ENABLE_CC_GRPC)
  set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
  find_package(Protobuf CONFIG REQUIRED)
  message(STATUS "Using protobuf ${Protobuf_VERSION}")
  include_directories(${Protobuf_INCLUDE_DIRS})
endif() # TRITON_ENABLE_CC_GRPC

#
# GRPC
#
if(TRITON_ENABLE_CC_GRPC)
  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION}")
  include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)
endif() # TRITON_ENABLE_CC_GRPC

add_subdirectory(triton_library)

add_subdirectory(backend)

# find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${OpenCV_LIBS}")
# message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Get user home
if(WIN32)
    set(USER_HOME $ENV{USERPROFILE})
else()
    set(USER_HOME $ENV{HOME})
endif()

# Print user home
message("User home directory: ${USER_HOME}")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION_STRING}")
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})
endif()



# find torch
set(CMAKE_PREFIX_PATH "${USER_HOME}/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)
message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")

set(CUDA_ARCHITECTURES 75)
set(USE_CUDNN 1)
set(USE_CUSPARSELT 1)
set(USE_CUDSS 1)
set(USE_CUFILE 1)

# find tbb
find_package(TBB REQUIRED)


add_subdirectory(frontend)
