#pragma once
#include "../inc/inc.hh"
#include "../common/logger.hh"
#include "../common/conf.hh"
#include "../common/cache.hh"
#include "../util/json/json.h"
#include "image_classify.hh"
#include "worker.hh"
#include "../common/concurrency_queue.hh"
#include <iostream>
#include <memory>
#include <string>

using elasticcdc::ElasticcdcReply;
using grpc::ServerWriter;

class Backend {
public:
    Backend(const std::string& _conf_path);
    ~Backend();

    void SetCache(const Json::Value& cache_config);
    void Exec(const ImageArgs& request);

private:
    std::shared_ptr<Config> conf_;
    std::shared_ptr<BasicCache<std::string, std::string>> cache_;

    std::shared_ptr<BatchWorker> rep_batch_worker_;
    std::shared_ptr<BatchWorker> cdc_batch_worker_;
    std::shared_ptr<InferWorker> infer_worker_;
    std::shared_ptr<ReplyWorker> reply_worker_;

    std::shared_ptr<SingleQueryQueue> rep_recv_queue_;
    std::shared_ptr<std::mutex> rep_recv_mutex_;
    std::shared_ptr<std::condition_variable> rep_recv_cv_;

    std::shared_ptr<SingleQueryQueue> cdc_recv_queue_;
    std::shared_ptr<std::mutex> cdc_recv_mutex_;
    std::shared_ptr<std::condition_variable> cdc_recv_cv_;
    
    std::shared_ptr<BatchQueryQueue> batch_queue_;
    std::shared_ptr<std::mutex> batch_mutex_;
    std::shared_ptr<std::condition_variable> batch_cv_;

    std::shared_ptr<BatchQueryQueue> infer_queue_;
    std::shared_ptr<std::mutex> infer_mutex_;
    std::shared_ptr<std::condition_variable> infer_cv_;

    void forwardBatchSize(int value);
    void backwardBatchSize(double value);
    bool ifAdjustBatch();
    void setAdjustBatch(bool value);
    bool ifFirstAdjust();
    void setFirstAdjust(bool value);
    int batch_size_1_;
    int batch_size_2_;
    bool batch_size_adjust;
    bool first_adjust;

};