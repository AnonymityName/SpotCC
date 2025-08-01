cmake_minimum_required(VERSION 3.17)

project(tritonclient LANGUAGES C CXX)

# Use C++17 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_ARCHITECTURES 75)
set(USE_CUDNN 1)
set(USE_CUSPARSELT 1)
set(USE_CUDSS 1)
set(USE_CUFILE 1)

#
# Options
#
set(TRITON_VERSION "0.0.0" CACHE STRING "Version for the clients")
set(PERF_ANALYZER_VERSION ${TRITON_VERSION} CACHE STRING "Build Version for Perf Analyzer")
option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)
option(TRITON_ENABLE_ZLIB "Include ZLIB library in build" ON)

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_THIRD_PARTY_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/third_party repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-third-party
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/third_party.git
  GIT_TAG ${TRITON_THIRD_PARTY_REPO_TAG}
  GIT_SHALLOW ON
)
set(TRITON_THIRD_PARTY_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third-party)
FetchContent_MakeAvailable(repo-third-party)

# Some libs are installed to ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib64 instead
# of ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib on Centos
set (LIB_DIR "lib")
# /etc/os-release does not exist on Windows
if(EXISTS "/etc/os-release")
  file(STRINGS /etc/os-release DISTRO REGEX "^NAME=")
  string(REGEX REPLACE "NAME=\"(.*)\"" "\\1" DISTRO "${DISTRO}")
  message(STATUS "Distro Name: ${DISTRO}")
  if(DISTRO MATCHES "CentOS.*")
    set (LIB_DIR "lib64")
  endif()
endif()

# Need to use ExternalProject for our builds so that we can get the
# correct dependencies between our components and the ExternalProject
# dependencies (found in the third_party repo)
include(ExternalProject)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRITON_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cc-clients/install)
else()
  set(TRITON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
if (OPENSSL_ROOT_DIR)
  set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
endif()

set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "")
if (CMAKE_TOOLCHAIN_FILE)
  set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "-DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}")
endif()

set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "")
if (VCPKG_TARGET_TRIPLET)
  set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "-DVCPKG_TARGET_TRIPLET:STRING=${VCPKG_TARGET_TRIPLET}")
endif()

# Location where protobuf-config.cmake will be installed varies by
# platform
if (WIN32)
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/cmake")
else()
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/${LIB_DIR}/cmake/protobuf")
endif()


set(_cc_client_depends re2)
set(_cc_client_depends ${_cc_client_depends} curl)
if(${TRITON_ENABLE_CC_GRPC})
set(_cc_client_depends ${_cc_client_depends} grpc protobuf)
endif() # TRITON_ENABLE_CC_GRPC


ExternalProject_Add(cc-clients
PREFIX cc-clients
SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cc-clients"
CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCURL_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/${LIB_DIR}/cmake/CURL
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -DgRPC_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
    -Dabsl_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/${LIB_DIR}/cmake/absl
    -Dre2_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/re2/${LIB_DIR}/cmake/re2
    -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/${LIB_DIR}/cmake/c-ares
    -DGTEST_ROOT:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/googletest
    -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
    -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
    -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
    -DPERF_ANALYZER_VERSION:STRING=${PERF_ANALYZER_VERSION}
    -DTRITON_ENABLE_CC_GRPC:BOOL=${TRITON_ENABLE_CC_GRPC}
    -DTRITON_ENABLE_ZLIB:BOOL=${TRITON_ENABLE_ZLIB}
    -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}
DEPENDS ${_cc_client_depends}
)

# add_subdirectory(src)