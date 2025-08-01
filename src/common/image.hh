#ifndef _IMAGE_HH_
#define _IMAGE_HH_

#include "../inc/inc.hh"
#include <grpcpp/grpcpp.h>


#include "../protocol/elasticcdc.grpc.pb.h"

struct ImageArgs {
  std::string model_name;
  std::string scale;
  std::string filename;
  std::string data;
  std::shared_ptr<grpcStream> stream;
  int id;
  int encode_id;
  std::string encode_type;
  bool is_parity_data_ = false;
  double cdc_infer_time_;
  double backup_infer_time_;
  double decode_time_;
  bool end_signal_ = false;
  bool is_recompute_ = false;
};

struct ImageClassifyArgs {
  std::string model_name;
  std::string scale;
  std::string filename;
  std::vector<uint8_t> data;
  std::shared_ptr<grpcStream> stream;
  int id;
  int encode_id;
  std::string encode_type;
  bool is_parity_data_ = false;
  bool end_signal_ = false;
  bool is_recompute_ = false;
};



class Image {
    std::size_t id = 0; //encode id
    cv::Mat img_data;

public:
    explicit Image();
    Image(const std::string filename, const std::string format);
    Image(const cv::Mat& img);

    // ~Image();

    void readImage(const std::string filename, const std::string format);
    void showImage();

    torch::Tensor* matToTensor();
    void matFromTensor(torch::Tensor*);
    std::vector<uint8_t> matToBytes();

    cv::Mat getImage() { return this -> img_data; };
    int getRows() { return this -> img_data.rows; };
    int getCols() { return this -> img_data.cols; };
    int getChannels() { return this -> img_data.channels(); };
    
};

#endif