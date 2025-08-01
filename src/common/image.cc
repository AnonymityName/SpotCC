#include "image.hh"

Image::Image(const std::string filename, const std::string format){
    if(format == "IMREAD_COLOR")
            img_data = cv::imread(filename,cv::IMREAD_COLOR);
    else if (format == "IMREAD_GRAYSCALE")
            img_data = cv::imread(filename,cv::IMREAD_GRAYSCALE);
    else if (format == "IMREAD_UNCHANGED")
            img_data = cv::imread(filename,cv::IMREAD_UNCHANGED);
    else {
        std::cout << "IMAGE FORMAT ERROR!" << std::endl; 
        exit(-1);  
    }
            
    if(img_data.empty()) {
        std::cout << "IMAGE EMPTY!" << std::endl;
    }
}


Image::Image(const cv::Mat& img) {
    if(img_data.empty()) {
        std::cerr << "IMAGE EMPTY!" << std::endl;
    }
    img.copyTo(this -> img_data);
}

/*
 * read the image from the file
*/
void Image::readImage(const std::string filename, const std::string format){
    if (format == "IMREAD_COLOR")
            img_data = cv::imread(filename,cv::IMREAD_COLOR);
    else if (format == "IMREAD_GRAYSCALE")
            img_data = cv::imread(filename,cv::IMREAD_GRAYSCALE);
    else if (format == "IMREAD_UNCHANGED")
            img_data = cv::imread(filename,cv::IMREAD_UNCHANGED);
    else {
        std::cout << "IMAGE FORMAT ERROR!" << std::endl; 
        exit(-1); 
    }
             
    if(img_data.empty()) {
        std::cout << "IMAGE EMPTY!" << std::endl;
    }
}

torch::Tensor* Image::matToTensor(){
    assert(img_data.empty());
    torch::Tensor* tensor = new torch::Tensor;
    torch::Tensor tmp = torch::from_blob(this -> img_data.data,
                                         {this -> img_data.rows, this -> img_data.cols, this -> img_data.channels()});
    *tensor = tmp;
    return tensor;
}

void Image::showImage() {
    assert(img_data.empty());
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::imshow("Image", this -> img_data);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

std::vector<uint8_t> Image::matToBytes() {
    std::vector<uint8_t> buffer;
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    // 设置JPEG质量，例如90，表示90%质量
    params.push_back(90);
    cv::imencode(".jpg", this -> img_data, buffer, params);
    return buffer;
}
