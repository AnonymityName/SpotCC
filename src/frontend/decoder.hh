#pragma once
#include "../inc/inc.hh"
#include "../common/conf.hh"
#include "../common/logger.hh"
#include <torch/script.h>

std::shared_ptr<torch::Tensor> vectorToTensor(const std::vector<std::vector<uint8_t>>& data, bool use_cuda);

class Decoder {
public:
    // virtual torch::Tensor* decode(torch::Tensor* _data) = 0;
    // virtual std::shared_ptr<torch::Tensor> decode(const std::shared_ptr<torch::Tensor>& _data) = 0;

    virtual void decode(const std::vector<std::vector<uint8_t>>& data, std::vector<uint8_t>& res) = 0;

    std::shared_ptr<Config> conf_;
    uint32_t k_;
};

class LinearDecoder: public Decoder {
public:
    LinearDecoder(std::shared_ptr<Config> conf);
    std::shared_ptr<torch::Tensor> decode(const std::shared_ptr<torch::Tensor>& _data);

    void decode(const std::vector<std::vector<uint8_t>>& data, std::vector<uint8_t>& res);

private:
    std::shared_ptr<torch::Tensor> vectorToTensor(const std::vector<std::vector<uint8_t>>& data, bool use_cuda);

    // torch::Tensor* decode(const torch::Tensor* _data);
};


class DistilledDecoder: public Decoder {
public:
    DistilledDecoder(std::shared_ptr<Config> _conf);
    ~DistilledDecoder();

    std::shared_ptr<torch::Tensor> decode(const std::shared_ptr<torch::Tensor>& _data);
    void decode(const std::vector<std::vector<uint8_t>>& data, std::vector<uint8_t>& res);
    // void run();

private:
    torch::jit::script::Module model;
    std::shared_ptr<torch::Tensor> vectorToTensor(const std::vector<std::vector<uint8_t>>& data, bool use_cuda);
};
