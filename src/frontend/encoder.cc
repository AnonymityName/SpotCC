#include "encoder.hh"

LinearEncoder::LinearEncoder(std::shared_ptr<Config> _conf) {
    conf = _conf;
    k = conf->k;
   
}

std::shared_ptr<torch::Tensor> LinearEncoder::encode(const std::shared_ptr<torch::Tensor>& _data) {
    assert(_data != nullptr);
    assert(_data->sizes().size() > 0);
    assert(_data->sizes()[0] == k);
    assert(k != 0);

    torch::Tensor t_sum = torch::sum(static_cast<int>(1/k)*(*_data), 0);
    return std::make_shared<torch::Tensor>(t_sum);
}

std::shared_ptr<torch::Tensor> LinearEncoder::vectorToTensor(const std::vector<std::vector<uint8_t>>& data) {
    assert(data.size() > 0);
    int rows = data.size();
    int cols = data[0].size();
    std::vector<float> flattened_data(rows*cols);
    int index = 0;
    for (const auto& row : data) {
        for (const auto ele : row) {
            flattened_data[index++] = static_cast<float>(ele);
        }
    }
    auto tensor = torch::from_blob(flattened_data.data(), {rows, cols}, torch::kFloat32).clone();

    if (conf->use_cuda) {
        assert(torch::cuda::is_available());
        tensor = tensor.to(torch::kCUDA);
    }

    return std::make_shared<torch::Tensor>(tensor);
}

void LinearEncoder::encode(const std::vector<std::vector<uint8_t>>& data, std::vector<uint8_t>& res){
    auto dataTensor = vectorToTensor(data);
    auto encodedTensor = encode(dataTensor);

    auto flattenedTensor = conf->use_cuda ? 
                            encodedTensor->flatten().to(torch::kUInt8).to(torch::kCPU):
                            encodedTensor->flatten().to(torch::kUInt8);

    uint8_t* dataPtr = flattenedTensor.data_ptr<uint8_t>();
    size_t numElements = flattenedTensor.numel();

    res.assign(dataPtr, dataPtr + numElements);
}

// ReplicaEncoder::ReplicaEncoder(std::shared_ptr<Config> _conf) {
//     conf = _conf;
//     back_num_ = conf->backup_num;

// }

// std::shared_ptr<torch::Tensor> ReplicaEncoder::encode(const std::shared_ptr<torch::Tensor>& _data) {
//     assert(_data->sizes().size() > 0);
//     assert(back_num_ > 0);

//     auto res = std::make_shared<torch::Tensor>(_data->clone());
//     return res;
// }

// void ReplicaEncoder::encode(const std::vector<std::vector<uint8_t>>& data, std::vector<uint8_t>& res) {
//     auto dataTensor = vectorToTensor(data);
//     auto t = dataTensor->clone();
//     auto encodedTensor = encode(dataTensor);
//     assert(encoderTensor != nullptr);
//     auto flattenedTensor = encodedTensor->flatten().to(torch::kUInt8);
//     uint8_t* dataPtr = flattenedTensor.data_ptr<uint8_t>();
//     size_t numElements = flattenedTensor.numel();
//     res.assign(dataPtr, dataPtr + numElements);
// }