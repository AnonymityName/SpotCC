# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 3.18)

find_package(Threads REQUIRED)
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
# find torch
# set(CMAKE_PREFIX_PATH "${USER_HOME}/libtorch" ${CMAKE_PREFIX_PATH})
# find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(_GRPC_GRPCPP gRPC::grpc++)

set(CUDA_ARCHITECTURES 75)
set(USE_CUDNN 1)
set(USE_CUSPARSELT 1)
set(USE_CUDSS 1)
set(USE_CUFILE 1)


#
# backend_server
#
find_package(OpenCV REQUIRED)
add_executable(
  backend_server
  backend_server.cc
  image_classify.cc
  backend.cc
  worker.cc
  ../common/cache.cc
  ../common/conf.cc
  ../util/jsoncpp.cpp
  ${hw_proto_srcs}
  ${hw_grpc_srcs}
  ../protocol/elasticcdc.grpc.pb.cc
  ../protocol/elasticcdc.grpc.pb.h
  ../protocol/elasticcdc.pb.cc
  ../protocol/elasticcdc.pb.h
  $<TARGET_OBJECTS:json-utils-library>
)
target_include_directories(
  backend_server
  PRIVATE ${OpenCV_INCLUDE_DIRS}
  PRIVATE ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(
  backend_server
  PRIVATE
    # elasticcdc_grpc_proto
    absl::check
    absl::flags
    absl::flags_parse
    absl::log
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF}
    grpcclient_static
    httpclient_static
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)
install(
  TARGETS backend_server
  RUNTIME DESTINATION bin
)

# 
# monitor
#
# add_executable(
#   monitor
#   monitor.cc
#   ../common/conf.cc
#   ../util/jsoncpp.cpp
#   ../protocol/elasticcdc.grpc.pb.cc
#   ../protocol/elasticcdc.grpc.pb.h
#   ../protocol/elasticcdc.pb.cc
#   ../protocol/elasticcdc.pb.h
# )
# target_link_libraries(
#   monitor
#   ${_REFLECTION}
#   ${_GRPC_GRPCPP}
#   ${_PROTOBUF_LIBPROTOBUF}
# )