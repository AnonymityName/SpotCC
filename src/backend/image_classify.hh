#pragma once
#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <opencv2/core/version.hpp>
#include <queue>
#include <string>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "http_client.h"
#include "grpc_client.h"
#include "json_utils.h"
#include "worker.hh"
#include "query.hh"
#if CV_MAJOR_VERSION == 2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#elif CV_MAJOR_VERSION >= 3
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

#if CV_MAJOR_VERSION == 4
#define GET_TRANSFORMATION_CODE(x) cv::COLOR_##x
#else
#define GET_TRANSFORMATION_CODE(x) CV_##x
#endif

namespace tc = triton::client;

namespace {

// enum ScaleType { NONE = 0, VGG = 1, INCEPTION = 2 };

enum ProtocolType { HTTP = 0, GRPC = 1 };

struct ModelInfo {
  std::string output_name_;
  std::vector<std::string> output_names_;
  std::string input_name_;
  std::string input_datatype_;
  // The shape of the input
  int input_c_;
  int input_h_;
  int input_w_;
  // The format of the input
  std::string input_format_;
  int type1_;
  int type3_;
  int max_batch_size_;
};

void Preprocess(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
            	size_t img_channels, const cv::Size& img_size, const ScaleType scale,
    			std::vector<uint8_t>* input_data); 
void Postprocess(const std::unique_ptr<tc::InferResult> result,
  const std::vector<std::string>& filenames, const size_t batch_size,
  const std::string& output_name, const size_t topk, const bool batching, std::vector<std::string>& res);



ScaleType ParseScale(const std::string& str);


ProtocolType ParseProtocol(const std::string& str);

bool ParseType(const std::string& dtype, int* type1, int* type3);


void ParseModelHttp(const rapidjson::Document& model_metadata,
    				const rapidjson::Document& model_config, const size_t batch_size,
    				ModelInfo* model_info);

void FileToInputData(const std::string& filename, size_t c, size_t h, size_t w,
    				const std::string& format, int type1, int type3, ScaleType scale,
    				std::vector<uint8_t>* input_data);

union TritonClient {
  TritonClient()
  {
    new (&http_client_) std::unique_ptr<tc::InferenceServerHttpClient>{};
  }
  ~TritonClient() {}

  std::unique_ptr<tc::InferenceServerHttpClient> http_client_;
  std::unique_ptr<tc::InferenceServerGrpcClient> grpc_client_;
};

}  // namespace



typedef struct ImageArgs {
  std::string model_name;
  std::string scale;
  std::string filename;
  std::string data;
  grpcStream* stream;
  int id;
  int encode_id;
  std::string encode_type;
  bool is_parity_data_;
  double cdc_infer_time;
  double backup_infer_time;
  double decode_time;
  int front_id;
  bool end_signal;
  bool recompute;
}ImageArgs;

typedef struct ImageClassifyArgs {
  std::string model_name;
  std::string scale;
  std::string filename;
  std::vector<uint8_t> data;
  grpcStream* stream;
  int id;
}ImageClassifyArgs;


// int ImageClassify(const ImageClassifyArgs& args, std::string& res);
int ImageClassify(BatchQuery& batchq, std::vector<std::string>& res);