#include "backend.hh"


/**
 * Backend
 * 
 */ 
Backend::Backend(const std::string& conf_path) {
    conf_ = std::make_shared<Config>(conf_path);
    conf_->parse();
    SetCache(conf_->getCacheConfig());
    batch_size_1_ = conf_->batch_size_1;
    batch_size_2_ = conf_->batch_size_2;
    batch_size_adjust = conf_->batch_mode == "auto" ? true : false;
    first_adjust = true;

    rep_recv_queue_ = std::make_shared<SingleQueryQueue>();
    cdc_recv_queue_ = std::make_shared<SingleQueryQueue>();
    batch_queue_ = std::make_shared<BatchQueryQueue>();
    infer_queue_ = std::make_shared<BatchQueryQueue>();

    rep_recv_mutex_ = std::make_shared<std::mutex>();
    cdc_recv_mutex_ = std::make_shared<std::mutex>();
    batch_mutex_ = std::make_shared<std::mutex>();
    infer_mutex_ = std::make_shared<std::mutex>();

    rep_recv_cv_ = std::make_shared<std::condition_variable>();
    cdc_recv_cv_ = std::make_shared<std::condition_variable>();
    batch_cv_ = std::make_shared<std::condition_variable>();
    infer_cv_ = std::make_shared<std::condition_variable>();

    rep_batch_worker_ = std::make_shared<BatchWorker>(conf_, batch_size_1_, rep_recv_queue_, rep_recv_mutex_, rep_recv_cv_, batch_queue_, batch_mutex_, batch_cv_);
    cdc_batch_worker_ = std::make_shared<BatchWorker>(conf_, batch_size_2_, cdc_recv_queue_, cdc_recv_mutex_, cdc_recv_cv_, batch_queue_, batch_mutex_, batch_cv_);
    infer_worker_ = std::make_shared<InferWorker>(conf_, batch_queue_, batch_mutex_, batch_cv_, infer_queue_, infer_mutex_, infer_cv_);
    reply_worker_ = std::make_shared<ReplyWorker>(conf_, infer_queue_, infer_mutex_, infer_cv_, nullptr, nullptr, nullptr);
    LOG_INFO("Backend created");
}

Backend::~Backend() {}

void Backend::SetCache(const Json::Value& cache_config) {
    const std::string strategy = cache_config.get("strategy", "").asString();
    if (strategy == "lru") {
        cache_ = std::make_shared<LruCache<std::string, std::string>>(cache_config);
    } else if (strategy == "lfu") {
        assert(false && "not implemented yet");
    } else {
        assert(false && "undefine cache strategy");
    }
}

bool Backend::ifAdjustBatch() {
    return batch_size_adjust;
}

bool Backend::ifFirstAdjust() {
    return first_adjust;
}

void Backend::setAdjustBatch(bool value) {
    batch_size_adjust = value;
}

void Backend::setFirstAdjust(bool value) {
    first_adjust = value;
}

void Backend::forwardBatchSize(int value) {
    batch_size_1_ += value;
    batch_size_2_ -= value;
}

void Backend::backwardBatchSize(double value) {
    batch_size_1_ *= (1 - conf_->dec_value);
    batch_size_2_ *= (1 + conf_->dec_value);
}

/**
 * 1. check cache
 * 2. if cache miss, transform request into query, push query into recv queue
 * 3. update cache
 */
void Backend::Exec(const ImageArgs& request) {
    // 1. check cache
    std::string reply_info;
    if (cache_->get(request.filename, reply_info)) {
        LOG_INFO("cache hit, request info: %s", request.filename.c_str());
        // reply
        ElasticcdcReply reply;
        reply.set_reply_info(reply_info);
        request.stream->Write(reply);
        return;
    }
    // 2. if cache miss, transform request into query, push query into recv queue
    LOG_INFO("backend exec request, filename: %s, model: %s, scale: %s", request.filename.c_str(), request.model_name.c_str(), request.scale.c_str());
    // assert(recv_mutex_ != nullptr);
    // std::cout << "1" << std::endl;
    // std::unique_lock<std::mutex> recv_lock(*recv_mutex_);
    // std::cout << "2" << std::endl;
    // ImageClassifyArgs request_info;
    // request_info.id = request.id;
    // request_info.scale = request.scale;
    // request_info.filename = request.filename;
    // request_info.data.assign(request.data.begin(), request.data.end());
    // request_info.stream = request.stream;

    // adjust bacth size based on the last latency
    if (request.cdc_infer_time > 0 && request.backup_infer_time > 0 && request.decode_time != 0 && ifAdjustBatch()) {
        // assert(request.backup_infer_time > 0);
        // assert(request.cdc_infer_time > 0);
        LOG_INFO("backup infer time: %lf, cdc infer time: %lf", request.backup_infer_time, request.cdc_infer_time);
        LOG_INFO("before ajust: backup batch size: %d, cdc batch size: %d", batch_size_1_, batch_size_2_);
        if (request.backup_infer_time < request.cdc_infer_time) {
            if (ifFirstAdjust())
                forwardBatchSize(conf_->inc_value);
            else {
                setAdjustBatch(false);
            }
        }
        else if (request.backup_infer_time >= request.cdc_infer_time) {
            backwardBatchSize(conf_->dec_value);
            if (ifFirstAdjust())
                setFirstAdjust(false);
        }
        else {
            setAdjustBatch(false);
        }
        LOG_INFO("after ajust: backup batch size: %d, cdc batch size: %d", batch_size_1_, batch_size_2_);
    }

    std::vector<uint8_t> data;
    data.assign(request.data.begin(), request.data.end());
    LOG_INFO("push query: %d to recv queue", request.id);
    auto query = new SingleQuery(request.model_name, 
                                    request.scale, 
                                    request.filename, 
                                    request.id, data, 
                                    request.encode_type, 
                                    request.stream, 
                                    request.front_id, 
                                    request.end_signal,
                                    request.recompute);
    std::cout << "query->encode_type_:" << query->encode_type_ << std::endl;
    std::cout << "query->end_signal:" << query->end_signal_ << std::endl;
    std::cout << "query->recompute_:" << query->recompute_ << std::endl;
    if (query->recompute_) {
        // std::cout << "recompute: push to rep queue" << std::endl;
        rep_recv_queue_->Push(query);
        rep_recv_cv_->notify_all();
    }
    else if (query->end_signal_) {
        rep_recv_queue_->Push(query);
        cdc_recv_queue_->Push(query);
        rep_recv_cv_->notify_all();
        cdc_recv_cv_->notify_all();
    }
    else if (query->encode_type_ == "Backup") {
        rep_recv_queue_->Push(query);
        rep_recv_cv_->notify_all();
    }
    else if (query->encode_type_ == "CDC") {
        // std::cout << "1" << std::endl;
        // std::cout << req
        cdc_recv_queue_->Push(query);
        cdc_recv_cv_->notify_all();
        // std::cout << "2" << std::endl;
    }
    // recv_lock.unlock(); 
    // notify batch_query_thread_(wait for batch queue full)

    // 3. update cache
    // cache_->put(request.filename, reply_info);
}
