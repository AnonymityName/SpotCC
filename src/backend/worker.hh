#pragma once
#include "../inc/inc.hh"
#include "../common/concurrency_queue.hh"
#include "../common/logger.hh"
#include "../common/conf.hh"
#include <iostream>
#include <memory>
#include <string>
#include "image_classify.hh"
#include "query.hh"

using grpc::ServerWriter;

using SingleQueryQueue = ConcurrencyQueue<SingleQuery*>;
using BatchQueryQueue = ConcurrencyQueue<BatchQuery*>;

// class Worker {
// public:
//     virtual void run() = 0;

//     std::shared_ptr<QueryQueue> queue_1_;
//     std::shared_ptr<std::mutex> mtx_1_;
//     std::shared_ptr<std::condition_variable> cv_1_;

//     std::shared_ptr<QueryQueue> queue_2_;
//     std::shared_ptr<std::mutex> mtx_2_;
//     std::shared_ptr<std::condition_variable> cv_2_;
// };

class BatchWorker {
public:
    BatchWorker(std::shared_ptr<Config> conf,
                int batch_size,
                std::shared_ptr<SingleQueryQueue> queue_1,
                std::shared_ptr<std::mutex> mtx_1,
                std::shared_ptr<std::condition_variable> cv_1,
                std::shared_ptr<BatchQueryQueue> queue_2,
                std::shared_ptr<std::mutex> mtx_2,
                std::shared_ptr<std::condition_variable> cv_2);
    ~BatchWorker();

    // void forwardBatchSize(int value);
    // void backwardBatchSize(double value);
    // bool ifAdjustBatch();
    // void setAdjustBatch(bool value);
    // bool ifFirstAdjust();
    // void setFirstAdjust(bool value);
    void setBatchSize(int value);

    std::thread batch_thread_;

private:
    void run();
    std::shared_ptr<SingleQueryQueue> queue_1_;
    std::shared_ptr<std::mutex> mtx_1_;
    std::shared_ptr<std::condition_variable> cv_1_;

    std::shared_ptr<BatchQueryQueue> queue_2_;
    std::shared_ptr<std::mutex> mtx_2_;
    std::shared_ptr<std::condition_variable> cv_2_;

    void BatchInputData();
    std::mutex batch_size_mtx_;
    std::condition_variable batch_size_cv_;
    int batch_size_;
    // bool batch_size_adjust;
    // bool first_adjust;
    std::shared_ptr<Config> conf_;
    BatchQuery* createBatchQuery(std::shared_ptr<SingleQueryQueue> queue, std::shared_ptr<std::mutex> mutex, 
        std::shared_ptr<std::condition_variable> cv, int batch_size);
};

class InferWorker {
public:
    InferWorker(std::shared_ptr<Config> conf,
                std::shared_ptr<BatchQueryQueue> queue_1,
                std::shared_ptr<std::mutex> mtx_1,
                std::shared_ptr<std::condition_variable> cv_1,
                std::shared_ptr<BatchQueryQueue> queue_2,
                std::shared_ptr<std::mutex> mtx_2,
                std::shared_ptr<std::condition_variable> cv_2);
    ~InferWorker();

    std::thread infer_thread_;

private:
    void run();
    std::shared_ptr<BatchQueryQueue> queue_1_;
    std::shared_ptr<std::mutex> mtx_1_;
    std::shared_ptr<std::condition_variable> cv_1_;

    std::shared_ptr<BatchQueryQueue> queue_2_;
    std::shared_ptr<std::mutex> mtx_2_;
    std::shared_ptr<std::condition_variable> cv_2_;
    
    std::shared_ptr<Config> conf_;
}; 

class ReplyWorker {
public:
    ReplyWorker(std::shared_ptr<Config> conf,
                std::shared_ptr<BatchQueryQueue> queue_1,
                std::shared_ptr<std::mutex> mtx_1,
                std::shared_ptr<std::condition_variable> cv_1,
                std::shared_ptr<BatchQueryQueue> queue_2,
                std::shared_ptr<std::mutex> mtx_2,
                std::shared_ptr<std::condition_variable> cv_2);
    ~ReplyWorker();

    std::thread reply_thread_;

private:
    void run();
    std::shared_ptr<BatchQueryQueue> queue_1_;
    std::shared_ptr<std::mutex> mtx_1_;
    std::shared_ptr<std::condition_variable> cv_1_;

    std::shared_ptr<BatchQueryQueue> queue_2_;
    std::shared_ptr<std::mutex> mtx_2_;
    std::shared_ptr<std::condition_variable> cv_2_;

    std::shared_ptr<Config> conf_;
}; 

class Ajustor {
public:
    Ajustor();

private:
    long time1_;
    long time2_;

    std::shared_ptr<ConcurrencyQueue<long>> queue_;
    std::shared_ptr<std::mutex> mtx_;
    std::shared_ptr<std::condition_variable> cv_;

};