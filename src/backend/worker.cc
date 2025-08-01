#include "worker.hh"

/**
 * BatchWorker
 * 
 */
BatchWorker::BatchWorker(std::shared_ptr<Config> conf,
                        int batch_size,
                        std::shared_ptr<SingleQueryQueue> queue_1,
                        std::shared_ptr<std::mutex> mtx_1,
                        std::shared_ptr<std::condition_variable> cv_1,
                        std::shared_ptr<BatchQueryQueue> queue_2,
                        std::shared_ptr<std::mutex> mtx_2,
                        std::shared_ptr<std::condition_variable> cv_2):
                        queue_1_(queue_1), 
                        mtx_1_(mtx_1),
                        cv_1_(cv_1),
                        queue_2_(queue_2),
                        mtx_2_(mtx_2),
                        cv_2_(cv_2),
                        conf_(conf)
{
    batch_size_ = batch_size;
    batch_thread_ = std::thread(&BatchWorker::run, this);
}

BatchWorker::~BatchWorker() {
    batch_thread_.join();
}

BatchQuery* BatchWorker::createBatchQuery(std::shared_ptr<SingleQueryQueue> queue, 
                                            std::shared_ptr<std::mutex> mutex, 
                                            std::shared_ptr<std::condition_variable> cv,
                                            int batch_size) {
    std::unique_lock<std::mutex> lock(*mutex);                               
    SingleQuery* query = queue->Pop();
    LOG_INFO("pop query: %d from recv queue", query->id_);

    std::vector<std::vector<uint8_t>> batch_data;
    std::vector<grpcStream*> streams;
    std::vector<int> ids;
    std::string model_name = query->model_name_;
    std::string scale = query->scale_;
    std::vector<std::string> filenames;
    std::string encode_type = query->encode_type_;
    int id = query->id_;

    batch_data.emplace_back(query->data_);
    streams.emplace_back(query->stream_);
    filenames.emplace_back(query->filename_);
    ids.emplace_back(query->id_);

    for (int i = 0; i < batch_size - 1; i++) {
        SingleQuery* query = queue->Pop();
        LOG_INFO("pop query: %d from recv queue", query->id_);
        batch_data.emplace_back(query->data_);
        streams.emplace_back(query->stream_);
        filenames.emplace_back(query->filename_);
        ids.emplace_back(query->id_);
    }
    
    auto batch_query = new BatchQuery(model_name, scale, filenames, id, batch_data, streams, encode_type, ids);
    lock.unlock();
    cv->notify_all();
    return batch_query;
}

// bool BatchWorker::ifAdjustBatch() {
//     return batch_size_adjust;
// }

// bool BatchWorker::ifFirstAdjust() {
//     return first_adjust;
// }

// void BatchWorker::setAdjustBatch(bool value) {
//     batch_size_adjust = value;
// }

// void BatchWorker::setFirstAdjust(bool value) {
//     first_adjust = value;
// }

// void BatchWorker::forwardBatchSize(int value) {
//     batch_size_1_ += value;
//     batch_size_2_ -= value;
// }

// void BatchWorker::backwardBatchSize(double value) {
//     batch_size_1_ *= (1-conf_->dec_value);
//     batch_size_2_ *= (1+conf_->dec_value);
// }

void BatchWorker::setBatchSize(int value) {
    std::unique_lock<std::mutex> lock(batch_size_mtx_);
    batch_size_ = value;
}

void BatchWorker::run() {
    while(true) {
        auto start = std::chrono::high_resolution_clock::now();
        {
            std::unique_lock<std::mutex> lock(*mtx_1_);
            LOG_INFO("BatchWorker waiting...");
            std::cout << "Batchsize: " << batch_size_ << std::endl;
            cv_1_->wait(lock, [this] { return (queue_1_->Size() >= batch_size_ || 
                                                queue_1_->Back()->end_signal_ ||
                                                queue_1_->Front()->recompute_); });
        }
        // pop query from recv queue
        // batch
        std::cout << "batch_size:" << batch_size_ << std::endl;
        if (queue_1_->Front()->recompute_) {
            std::cout << "receive recompute 1" << std::endl;
            int recompute_batch_size = 1;
            auto batch_query = createBatchQuery(queue_1_, mtx_1_, cv_1_, recompute_batch_size);
            std::unique_lock<std::mutex> batch_lock(*mtx_2_);
            queue_2_->Push(batch_query);
            LOG_INFO("push query: %d to batch queue", batch_query->id_);
            batch_lock.unlock();
            cv_2_->notify_all();
        }
        else if (queue_1_->Back()->end_signal_) {
            std::cout << "receive end signal 1" << std::endl;
            int last_batch_size_1 = queue_1_->Size()-1;
            if (last_batch_size_1 != 0) {
                auto batch_query = createBatchQuery(queue_1_, mtx_1_, cv_1_, last_batch_size_1);
                std::unique_lock<std::mutex> batch_lock(*mtx_2_);
                queue_2_->Push(batch_query);
                LOG_INFO("push query: %d to batch queue", batch_query->id_);
                batch_lock.unlock();
                cv_2_->notify_all();
            }
            queue_1_->Pop();
            std::cout << "receive end signal 2" << std::endl;
        }
        else if (queue_1_->Size() >= batch_size_) {
            auto batch_query = createBatchQuery(queue_1_, mtx_1_, cv_1_, batch_size_);
            std::unique_lock<std::mutex> batch_lock(*mtx_2_);
            queue_2_->Push(batch_query);
            LOG_INFO("push query: %d to batch queue", batch_query->id_);
            batch_lock.unlock();
            cv_2_->notify_all();
        }
        // SingleQuery* query = queue_1_->Pop();
        // LOG_INFO("pop query: %d from recv queue", query->id_);

        // std::vector<std::vector<uint8_t>> batch_data;
        // std::vector<grpcStream*> streams;
        // std::vector<int> ids;
        // std::string model_name = query->model_name_;
        // std::string scale = query->scale_;
        // std::vector<std::string> filenames;
        // // std::string filename = query->filename_;
        // std::string encode_type = query->encode_type_;
        // int id = query->id_;

        // batch_data.emplace_back(query->data_);
        // streams.emplace_back(query->stream_);
        // filenames.emplace_back(query->filename_);
        // ids.emplace_back(query->id_);

        // for (int i = 0; i < batch_size_-1; i++) {
        //     SingleQuery* query = queue_1_->Pop();
        //     LOG_INFO("pop query: %d from recv queue", query->id_);
        //     batch_data.emplace_back(query->data_);
        //     streams.emplace_back(query->stream_);
        //     filenames.emplace_back(query->filename_);
        //     ids.emplace_back(query->id_);
        // }
        
        // // push batch query into batch queue
        // auto batch_query = new BatchQuery(model_name, scale, filenames, id, batch_data, streams, encode_type, ids);

        // std::unique_lock<std::mutex> batch_lock(*mtx_2_);
        // queue_2_->Push(batch_query);
        // LOG_INFO("push query: %d to batch queue", id);
        // batch_lock.unlock();
        // cv_2_->notify_all();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                            (end_time - start).count();
        std::cout << "Batch Worker time: " << duration_time << std::endl;
    }
}

/**
 * InferWorker
 * 
 */
InferWorker::InferWorker(std::shared_ptr<Config> conf,
                        std::shared_ptr<BatchQueryQueue> queue_1,
                        std::shared_ptr<std::mutex> mtx_1,
                        std::shared_ptr<std::condition_variable> cv_1,
                        std::shared_ptr<BatchQueryQueue> queue_2,
                        std::shared_ptr<std::mutex> mtx_2,
                        std::shared_ptr<std::condition_variable> cv_2):
                        queue_1_(queue_1), 
                        mtx_1_(mtx_1),
                        cv_1_(cv_1),
                        queue_2_(queue_2),
                        mtx_2_(mtx_2),
                        cv_2_(cv_2),
                        conf_(conf)
{
    infer_thread_ = std::thread(&InferWorker::run, this);
}

InferWorker::~InferWorker() {
    infer_thread_.join();
}

void InferWorker::run() {
    while(true) {
        auto start1 = std::chrono::high_resolution_clock::now();
        {
            std::unique_lock<std::mutex> lock(*mtx_1_);
            LOG_INFO("InferWorker waiting...");
            cv_1_->wait(lock, [this] { return queue_1_->Size() > 0; });
        }
        
        auto batch_query = queue_1_->Pop();
        LOG_INFO("pop query: %d from batch queue", batch_query->id_);
        
        auto start = std::chrono::high_resolution_clock::now();
        // exec image classify task
        ImageClassify(*batch_query, batch_query->reply_info_);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        LOG_INFO("Infer time: %ld ms", duration);

        // push query into infer queue
        std::unique_lock<std::mutex> infer_lock(*mtx_2_);
        queue_2_->Push(batch_query);
        LOG_INFO("push query: %d to infer queue", batch_query->id_);
        infer_lock.unlock();
        cv_2_->notify_all();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                            (end_time - start1).count();
        std::cout << "Infer Worker time: " << duration_time << std::endl;
    }
}

/**
 * ReplyWorker
 * 
 */
ReplyWorker::ReplyWorker(std::shared_ptr<Config> conf,
                        std::shared_ptr<BatchQueryQueue> queue_1,
                        std::shared_ptr<std::mutex> mtx_1,
                        std::shared_ptr<std::condition_variable> cv_1,
                        std::shared_ptr<BatchQueryQueue> queue_2,
                        std::shared_ptr<std::mutex> mtx_2,
                        std::shared_ptr<std::condition_variable> cv_2):
                        queue_1_(queue_1), 
                        mtx_1_(mtx_1),
                        cv_1_(cv_1),
                        queue_2_(queue_2),
                        mtx_2_(mtx_2),
                        cv_2_(cv_2),
                        conf_(conf)
{
    reply_thread_ = std::thread(&ReplyWorker::run, this);
}

ReplyWorker::~ReplyWorker() {
    reply_thread_.join();
}

void ReplyWorker::run() {
    while(true) {
        auto start1 = std::chrono::high_resolution_clock::now();
        std::unique_lock<std::mutex> lock(*mtx_1_);
        LOG_INFO("ReplyWorker waiting...");
        cv_1_->wait(lock, [this] { return queue_1_->Size() > 0; });

        auto batch_query = queue_1_->Pop();
        LOG_INFO("pop query: %d from infer queue", batch_query->id_);
        
        assert(batch_query->batch_size_ == batch_query->data_.size());
        assert(batch_query->batch_size_ == batch_query->streams_.size());
        assert(batch_query->batch_size_ == batch_query->reply_info_.size());
        assert(batch_query->batch_size_ == batch_query->ids_.size());

        // reply to frontend
        for(int i = 0; i < batch_query->batch_size_; i++) {
            elasticcdc::ElasticcdcReply reply;
            reply.set_id(batch_query->ids_[i]);
            reply.set_reply_info(batch_query->reply_info_[i]);
            LOG_INFO("reply_info_size: %ld", batch_query->reply_info_[i].size());
            batch_query->streams_[i]->Write(reply);
            LOG_INFO("send query: %d to client", batch_query->ids_[i]);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                            (end_time - start1).count();
        std::cout << "Reply Worker time: " << duration_time << std::endl;
    }
}
