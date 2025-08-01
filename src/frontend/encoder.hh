#pragma once

#include "../inc/inc.hh"
#include "../common/conf.hh"

class Encoder {
public:
    uint32_t k;
    uint32_t back_num_;
    std::shared_ptr<Config> conf;
    virtual void encode(const std::vector<std::vector<uint8_t>>& data, std::vector<uint8_t>& res) = 0;
};

class LinearEncoder: public Encoder {
public:
    LinearEncoder(std::shared_ptr<Config> _conf);

    std::shared_ptr<torch::Tensor> vectorToTensor(const std::vector<std::vector<uint8_t>>& data);

    std::shared_ptr<torch::Tensor> encode(const std::shared_ptr<torch::Tensor>& _data);

    void encode(const std::vector<std::vector<uint8_t>>& data, std::vector<uint8_t>& res);

private: 
    
};

// class ReplicaEncoder: public Encoder {
// public:
//     ReplicaEncoder(std::shared_ptr<Config> _conf);

//     std::shared_ptr<torch::Tensor> encode(const std::shared_ptr<torch::Tensor>& _data);

//     void encode(const std::vector<std::vector<uint8_t>>& data, std::vector<uint8_t>& res);

// private:
// };