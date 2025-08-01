#pragma once
#include "../inc/inc.hh"
#include "encoder.hh"
#include "decoder.hh"
#include "../common/conf.hh"
#include "../common/image.hh"
// #include "../util/queue.hh"
#include "../protocol/elasticcdc.pb.h"
#include "../protocol/elasticcdc.grpc.pb.h"
#include <google/protobuf/empty.pb.h>
#include "Worker.hh"
// #include "monitor.hh"
#include "monitor2parts.hh"
#include "filter.hh"


class Frontend {
public:
    // Frontend(std::shared_ptr<Config> conf);
    Frontend(std::shared_ptr<Config> conf, std::shared_ptr<Monitor> monitor, std::shared_ptr<Filter> filter);
    ~Frontend();

    /**
     * @brief do process
     */
    void Exec(const ImageArgs& request_info);


private:
    std::uint32_t frontend_id = 0;

    std::shared_ptr<Config> conf_;

    std::shared_ptr<PreprocessWorker>pp_worker_;
    std::shared_ptr<EncodeWorker> encode_worker_;
    std::shared_ptr<InferWorker> infer_worker_;
    std::shared_ptr<DecodeWorker> decode_worker_;

    std::shared_ptr<QueryQueue> recv_queue_;
    std::shared_ptr<std::mutex> recv_mutex_;
    std::shared_ptr<std::condition_variable> recv_cv_;
    
    std::shared_ptr<QueryQueue> pp_queue_;
    std::shared_ptr<std::mutex> pp_mutex_;
    std::shared_ptr<std::condition_variable> pp_cv_;

    std::shared_ptr<QueryQueue> encode_queue_;
    std::shared_ptr<std::mutex> encode_mutex_;
    std::shared_ptr<std::condition_variable> encode_cv_;

    std::shared_ptr<QueryQueue> infer_queue_;
    std::shared_ptr<std::mutex> infer_mutex_;
    std::shared_ptr<std::condition_variable> infer_cv_;

    std::shared_ptr<Filter> filter_;

};

