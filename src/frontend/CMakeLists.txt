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

set(CMAKE_PREFIX_PATH "${USER_HOME}/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CUDA_ARCHITECTURES 75)
set(USE_CUDNN 1)
set(USE_CUSPARSELT 1)
set(USE_CUDSS 1)
set(USE_CUFILE 1)

set(_GRPC_GRPCPP gRPC::grpc++)

aux_source_directory(. DIR_LIB_SRCS)

include_directories(/usr/include/eigen3)

# message(${DIR_LIB_SRCS})
add_library(frontend ${DIR_LIB_SRCS})

target_link_libraries(
    frontend
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS}
)

add_executable(
    image_frontend
    ../protocol/elasticcdc.grpc.pb.cc
    ../protocol/elasticcdc.grpc.pb.h
    ../protocol/elasticcdc.pb.cc
    ../protocol/elasticcdc.pb.h
    ../common/conf.cc
    ../common/image.cc
    ../util/jsoncpp.cpp
    ../example/image_frontend.cc
    ../util/ARIMA/ARIMA.cc
)

target_link_libraries(
    image_frontend
    PRIVATE
        absl::check
        absl::flags
        absl::flags_parse
        absl::log
        frontend
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        TBB::tbb
)

install(
  TARGETS image_frontend
  RUNTIME DESTINATION bin
)

add_executable(
    image_client
    # ${hw_proto_srcs}
    # ${hw_grpc_srcs}
    ../protocol/elasticcdc.grpc.pb.cc
    ../protocol/elasticcdc.grpc.pb.h
    ../protocol/elasticcdc.pb.cc
    ../protocol/elasticcdc.pb.h
    ../common/conf.cc
    ../common/image.cc
    ../util/jsoncpp.cpp
    ../example/image_client.cc
)

target_include_directories(
    image_client PUBLIC
    ${PROTOBUF_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
)

target_link_libraries(
    image_client
    PRIVATE
        absl::check
        absl::flags
        absl::flags_parse
        absl::log
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        
        ${_PROTOBUF_LIBPROTOBUF}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
)


install(
  TARGETS image_client
  RUNTIME DESTINATION bin
)