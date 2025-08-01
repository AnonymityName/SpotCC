#pragma once
#include "../inc/inc.hh"

#if CV_MAJOR_VERSION == 4
#define GET_TRANSFORMATION_CODE(x) cv::COLOR_##x
#else
#define GET_TRANSFORMATION_CODE(x) CV_##x
#endif

class Preprocessor {
public:
    /**
     * @brief process the images: change the img channels ,resize the image and so on,
     *                            output is the input_data
     * @param format FORMAT_NHWC or FORMAT_NCHW
     * @param scale NONE , INCEPTION or VGG 
     */
    void Preprocess(
        const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
        size_t img_channels, const cv::Size& img_size, const ScaleType scale,
        std::vector<uint8_t>* input_data);

    /**
     * @brief Analysis of scale type
     **/
    ScaleType ParseScale(const std::string& str);

    /**
     * @brief Analysis of image type
     **/
    bool ParseType(const std::string& dtype, int* type1, int* type3);

    
};
