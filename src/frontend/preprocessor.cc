#include "preprocessor.hh"

void Preprocessor::Preprocess(
    const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size, const ScaleType scale,
    std::vector<uint8_t>* input_data)
{
  // Image channels are in BGR order. Currently model configuration
  // data doesn't provide any information as to the expected channel
  // orderings (like RGB, BGR). We are going to assume that RGB is the
  // most likely ordering and so change the channels to that ordering.

// 1. change the image color
  cv::Mat sample;
  if ((img.channels() == 3) && (img_channels == 1)) {
    cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGR2GRAY));
  } else if ((img.channels() == 4) && (img_channels == 1)) {
    cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGRA2GRAY));
  } else if ((img.channels() == 3) && (img_channels == 3)) {
    cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGR2RGB));
  } else if ((img.channels() == 4) && (img_channels == 3)) {
    cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGRA2RGB));
  } else if ((img.channels() == 1) && (img_channels == 3)) {
    cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(GRAY2RGB));
  } else if ((img.channels() == 1) && (img_channels == 1)) {
    sample = img.clone();
  } else {
    std::cerr << "unexpected number of channels " << img.channels()
              << " in input image, model expects " << img_channels << "."
              << std::endl;
    exit(1);
  }

  // 2. change the image size
  cv::Mat sample_resized;
  if (sample.size() != img_size) {
    cv::resize(sample, sample_resized, img_size);
  } else {
    sample_resized = sample;
  }

 // 3. change the image channels
  cv::Mat sample_type;
  sample_resized.convertTo(
      sample_type, (img_channels == 3) ? img_type3 : img_type1);
 
 // 4. deal with the image depend on the scene
  cv::Mat sample_final;
  if (scale == ScaleType::INCEPTION) {
    if (img_channels == 1) {
      sample_final = sample_type.mul(cv::Scalar(1 / 127.5));
      sample_final = sample_final - cv::Scalar(1.0);
    } else {
      sample_final =
          sample_type.mul(cv::Scalar(1 / 127.5, 1 / 127.5, 1 / 127.5));
      sample_final = sample_final - cv::Scalar(1.0, 1.0, 1.0);
    }
  } else if (scale == ScaleType::VGG) {
    if (img_channels == 1) {
      sample_final = sample_type - cv::Scalar(128);
    } else {
      sample_final = sample_type - cv::Scalar(123, 117, 104);
    }
  } else {
    sample_final = sample_type;
  }

  // Allocate a buffer to hold all image elements.
  size_t img_byte_size = sample_final.total() * sample_final.elemSize();
  size_t pos = 0;
  input_data->resize(img_byte_size);

  // For NHWC format Mat is already in the correct order but need to
  // handle both cases of data being contiguous or not.
  if (format.compare("FORMAT_NHWC") == 0) {
    if (sample_final.isContinuous()) {
      memcpy(&((*input_data)[0]), sample_final.datastart, img_byte_size);
      pos = img_byte_size;
    } else {
      size_t row_byte_size = sample_final.cols * sample_final.elemSize();
      for (int r = 0; r < sample_final.rows; ++r) {
        memcpy(
            &((*input_data)[pos]), sample_final.ptr<uint8_t>(r), row_byte_size);
        pos += row_byte_size;
      }
    }
  } else {
    // (format.compare("FORMAT_NCHW") == 0)
    //
    // For CHW formats must split out each channel from the matrix and
    // order them as BBBB...GGGG...RRRR. To do this split the channels
    // of the image directly into 'input_data'. The BGR channels are
    // backed by the 'input_data' vector so that ends up with CHW
    // order of the data.
    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < img_channels; ++i) {
      input_bgr_channels.emplace_back(
          img_size.height, img_size.width, img_type1, &((*input_data)[pos]));
      pos += input_bgr_channels.back().total() *
             input_bgr_channels.back().elemSize();
    }

    cv::split(sample_final, input_bgr_channels);
  }

  if (pos != img_byte_size) {
    std::cerr << "unexpected total size of channels " << pos << ", expecting "
              << img_byte_size << std::endl;
    exit(1);
  }
}

ScaleType Preprocessor::ParseScale(const std::string& str)
{
  if (str == "NONE") {
    return ScaleType::NONE;
  } else if (str == "INCEPTION") {
    return ScaleType::INCEPTION;
  } else if (str == "VGG") {
    return ScaleType::VGG;
  }

  std::cerr << "unexpected scale type \"" << str
            << "\", expecting NONE, INCEPTION or VGG" << std::endl;
  exit(1);

  return ScaleType::NONE;
}

bool Preprocessor::ParseType(const std::string& dtype, int* type1, int* type3)
{
  if (dtype.compare("UINT8") == 0) {
    *type1 = CV_8UC1;
    *type3 = CV_8UC3;
  } else if (dtype.compare("INT8") == 0) {
    *type1 = CV_8SC1;
    *type3 = CV_8SC3;
  } else if (dtype.compare("UINT16") == 0) {
    *type1 = CV_16UC1;
    *type3 = CV_16UC3;
  } else if (dtype.compare("INT16") == 0) {
    *type1 = CV_16SC1;
    *type3 = CV_16SC3;
  } else if (dtype.compare("INT32") == 0) {
    *type1 = CV_32SC1;
    *type3 = CV_32SC3;
  } else if (dtype.compare("FP32") == 0) {
    *type1 = CV_32FC1;
    *type3 = CV_32FC3;
  } else if (dtype.compare("FP64") == 0) {
    *type1 = CV_64FC1;
    *type3 = CV_64FC3;
  } else {
    return false;
  }

  return true;
}
