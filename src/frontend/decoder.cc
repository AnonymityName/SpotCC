#include "decoder.hh"

std::vector<float> StringToFloatVector(const std::string& str) {
  size_t float_count = str.size() / sizeof(float);
  const float* float_data = reinterpret_cast<const float*>(str.data());
  return std::vector<float>(float_data, float_data + float_count);
}

void simlutateDecodingProcess(std::string model_name) {
    std::this_thread::sleep_for(std::chrono::milliseconds(PROCESS_TIME.at(model_name)));
}

LinearDecoder::LinearDecoder(std::shared_ptr<Config> conf) {
    conf_ = conf;
    k_ = conf->k;
}

std::shared_ptr<torch::Tensor> LinearDecoder::decode(const std::shared_ptr<torch::Tensor>& _data) {
    assert(_data->sizes().size() > 0);
    assert(_data->sizes()[0] == k_);

    auto res = std::make_shared<torch::Tensor>();
    int len = _data->sizes()[0];

    *res = (*_data)[len-1] * static_cast<int>(k_); 

    torch::Tensor tmp = (*_data).slice(0, 0, (*_data).size(0) - 1);
    torch::Tensor sumTmp = torch::sum(tmp, 0);
    *res = *res - sumTmp;
    return res;
}

std::shared_ptr<torch::Tensor> LinearDecoder::vectorToTensor(const std::vector<std::vector<uint8_t>>& data, bool use_cuda) {
    assert(data.size() > 0);
    int rows = data.size();
    int cols = data[0].size();
    std::vector<float> flattened_data(rows*cols);
    // int index = 0;
    // for (const auto& row : data) {
    //     for (const auto ele : row) {
    //         flattened_data[index++] = static_cast<float>(ele);
    //     }
    // }
    float* dst = flattened_data.data();
    for (const auto& row : data) {
        for (uint8_t val : row) {
            *dst++ = static_cast<float>(val);
        }
    }
    auto tensor = torch::from_blob(flattened_data.data(), {rows, cols}, torch::kFloat32).clone();
    if (use_cuda) {
        assert(torch::cuda::is_available());
        tensor = tensor.to(torch::kCUDA);
    }

    return std::make_shared<torch::Tensor>(tensor);
}

void LinearDecoder::decode(const std::vector<std::vector<uint8_t>>& data, std::vector<uint8_t>& res){
    if (conf_->decoder_simulate) {
        simlutateDecodingProcess(conf_->model_name);
    }
    else {
        auto dataTensor = vectorToTensor(data, conf_->use_cuda);
        auto encodedTensor = decode(dataTensor);
        auto flattenedTensor = conf_->use_cuda ? 
                                encodedTensor->flatten().to(torch::kUInt8).to(torch::kCPU):
                                encodedTensor->flatten().to(torch::kUInt8);
                                
        uint8_t* dataPtr = flattenedTensor.data_ptr<uint8_t>();
        size_t numElements = flattenedTensor.numel();

        res.assign(dataPtr, dataPtr + numElements);
    }
    
}


DistilledDecoder::DistilledDecoder(std::shared_ptr<Config> conf) {
    conf_ = conf;
    k_ = conf->k;

    assert(conf_->decoder_ckpt != "null");

    // torch::set_num_threads(4);
    // torch::set_num_interop_threads(1);

    try{
        model = torch::jit::load(conf->decoder_ckpt, torch::kCUDA);
    } catch (const c10::Error& e) {
        // LOG_ERROR("error to load decoder model: %s", e.what());
        std::cerr << e.what() << std::endl;
    } catch (const std::exception& e) {
        // LOG_ERROR("error to load decoder model: %s", e.what());
        std::cerr << e.what() << std::endl;
    } 

    
}

DistilledDecoder::~DistilledDecoder() {}

std::shared_ptr<torch::Tensor> DistilledDecoder::vectorToTensor(const std::vector<std::vector<uint8_t>>& data, bool use_cuda) {
    int k = data.size();
    int c, h, w;
    std::vector<uint32_t> shape = DATASETS.at(conf_->model_name).second;
    c = shape[0];
    h = shape[1];
    w = shape[2];

    c *= k;

    int data_rows = data.size();
    int data_cols = data[0].size();
    // std::vector<float> flattened_data(data_rows*data_cols);

    std::vector<float> flattened_data(c * h * w, 0.0f);
    size_t idx = 0;
    for (const auto& row : data) {
        for (uint8_t val : row) {
            if (idx < flattened_data.size())
                flattened_data[idx++] = static_cast<float>(val);
        }
    }

    auto tensor = torch::from_blob(flattened_data.data(), {c, h, w}, torch::kFloat32).clone();
    
    if (use_cuda) {
        assert(torch::cuda::is_available());
        tensor = tensor.to(torch::kCUDA);
    }

    return std::make_shared<torch::Tensor>(tensor);
}

std::shared_ptr<torch::Tensor> DistilledDecoder::decode(const std::shared_ptr<torch::Tensor>& _data) {
    assert(torch::cuda::is_available());

    torch::IntArrayRef shape = _data->sizes();

    // std::cout << "_data shape: ";
    // for (auto size : shape) {
    //     std::cout << size << " ";
    // }
    // std::cout << std::endl;

    torch::Tensor new_tensor = _data->unsqueeze(0);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(new_tensor);

     try {
        if (conf_->model_name.find("irevnet") != std::string::npos) {
            auto output = model.forward(inputs).toTuple();
            auto first_tensor = output->elements()[0].toTensor();
            return std::make_shared<torch::Tensor>(first_tensor);
        }
        else {
            auto output = model.forward(inputs).toTensor();
            return std::make_shared<torch::Tensor>(output);
        }
    } catch (const torch::Error& e) {
        std::cerr << "Error during model forward pass: " << e.what() << std::endl;
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "General exception: " << e.what() << std::endl;
        return nullptr;
    }
}

void DistilledDecoder::decode(const std::vector<std::vector<uint8_t>>& data, std::vector<uint8_t>& res) {
    if (conf_->decoder_simulate) {
        simlutateDecodingProcess(conf_->model_name);
    }
    else{
        auto dataTensor = vectorToTensor(data, conf_->use_cuda);
        auto encodedTensor = decode(dataTensor);

        auto flattenedTensor = conf_->use_cuda ? 
                                encodedTensor->flatten().to(torch::kUInt8).to(torch::kCPU):
                                encodedTensor->flatten().to(torch::kUInt8);
                        
        uint8_t* dataPtr = flattenedTensor.data_ptr<uint8_t>();
        size_t numElements = flattenedTensor.numel();

        res.assign(dataPtr, dataPtr + numElements);
    }
    
}

// void DistilledDecoder::run() {

// }