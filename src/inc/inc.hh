#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <thread>
#include <set>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <memory>
#include <algorithm>
#include <sstream>
#include <random>
#include <utility>
#include <cstdint>
#include<unordered_set>


#include <cassert>
#include <cstring>
#include <ctime>

#include <torch/script.h>
#include <torch/torch.h>

// #if CV_MAJOR_VERSION == 2
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/opencv.hpp>
// #elif CV_MAJOR_VERSION >= 3
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
// #endif

#include <grpcpp/grpcpp.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#ifdef BAZEL_BUILD
#include "examples/protos/elasticcdc.grpc.pb.h"
#else
#include "../protocol/elasticcdc.grpc.pb.h"
#endif

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;
using elasticcdc::ElasticcdcRequest;
using elasticcdc::ElasticcdcReply;
using elasticcdc::ElasticcdcService;

using grpcStream = grpc::ServerReaderWriter<ElasticcdcReply, ElasticcdcRequest>;
using grpcStreamClient = grpc::ClientReaderWriter<ElasticcdcRequest, ElasticcdcReply>;

enum ScaleType { NONE = 0, VGG = 1, INCEPTION = 2 };

enum EncodeType {Backup = 0, CDC = 1};

enum NodeFlag {INVULNERABLE = 0, VULNERABLE = 1};


const std::unordered_map<std::string, std::pair<uint32_t, std::vector<uint32_t>>> DATASETS = {
    {"mnist-irevnet", {512 * 7 * 7 * 4, {8, 56, 56}}},
    {"fashion-irevnet", {512 * 7 * 7 * 4, {8, 56, 56}}},
    {"cifar10-irevnet", {512 * 8 * 8 * 4, {8, 64, 64}}},
    {"cifar10-resnet18", {10 * 4, {10}}},
    {"cifar10-resnet50", {10 * 4, {8, 128, 128}}},
    {"cifar10-vgg16", {10 * 4, {8, 128, 128}}},
    {"cifar10-vitbase", {10 * 4, {8, 128, 128}}},
    {"cifar10-vitlarge", {10 * 4, {8, 128, 128}}},
    {"cifar10-vithuge", {10 * 4, {8, 128, 128}}}
};

const std::unordered_map<std::string, size_t> PROCESS_TIME = {
    {"cifar10-irevnet", 20},
    {"cifar10-resnet50", 20},
    {"cifar10-vitbase", 20},
    {"cifar10-vgg16", 40},
    {"cifar10-vitlarge", 60},
    {"cifar10-vithuge", 100}
};