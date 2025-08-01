#include "frontend.hh"

// Frontend::Frontend(std::shared_ptr<Config> conf) {
//     conf_ = conf;

//     recv_queue_ = std::make_shared<QueryQueue>();
//     pp_queue_ = std::make_shared<QueryQueue>();
//     encode_queue_ = std::make_shared<QueryQueue>();
//     infer_queue_ = std::make_shared<QueryQueue>();

//     recv_mutex_ = std::make_shared<std::mutex>();
//     pp_mutex_ = std::make_shared<std::mutex>();
//     encode_mutex_ = std::make_shared<std::mutex>();
//     infer_mutex_ = std::make_shared<std::mutex>();

//     recv_cv_ = std::make_shared<std::condition_variable>();
//     pp_cv_ = std::make_shared<std::condition_variable>();
//     encode_cv_ = std::make_shared<std::condition_variable>();
//     infer_cv_ = std::make_shared<std::condition_variable>();

//     pp_worker_ = std::make_shared<PreprocessWorker>(conf_, recv_queue_, recv_mutex_, recv_cv_, pp_queue_, pp_mutex_, pp_cv_);
//     encode_worker_ = std::make_shared<EncodeWorker>(conf_, pp_queue_, pp_mutex_, pp_cv_, encode_queue_, encode_mutex_, encode_cv_);
//     infer_worker_ = std::make_shared<InferWorker>(conf_, encode_queue_, encode_mutex_, encode_cv_, infer_queue_, infer_mutex_, infer_cv_);
//     decode_worker_ = std::make_shared<DecodeWorker>(conf_, infer_queue_, infer_mutex_, infer_cv_, nullptr, nullptr, nullptr);
//     LOG_INFO("Frontend created");
// }

Frontend::Frontend(std::shared_ptr<Config> conf, std::shared_ptr<Monitor> monitor, std::shared_ptr<Filter> filter) {
    conf_ = conf;

    frontend_id = conf_ -> frontend_id; // need to complete!!!!!

    recv_queue_ = std::make_shared<QueryQueue>();
    pp_queue_ = std::make_shared<QueryQueue>();
    encode_queue_ = std::make_shared<QueryQueue>();
    infer_queue_ = std::make_shared<QueryQueue>();

    recv_mutex_ = std::make_shared<std::mutex>();
    pp_mutex_ = std::make_shared<std::mutex>();
    encode_mutex_ = std::make_shared<std::mutex>();
    infer_mutex_ = std::make_shared<std::mutex>();

    recv_cv_ = std::make_shared<std::condition_variable>();
    pp_cv_ = std::make_shared<std::condition_variable>();
    encode_cv_ = std::make_shared<std::condition_variable>();
    infer_cv_ = std::make_shared<std::condition_variable>();

    filter_ = filter;

    pp_worker_ = std::make_shared<PreprocessWorker>(conf_, recv_queue_, recv_mutex_, recv_cv_, pp_queue_, pp_mutex_, pp_cv_, monitor);
    encode_worker_ = std::make_shared<EncodeWorker>(conf_, pp_queue_, pp_mutex_, pp_cv_, encode_queue_, encode_mutex_, encode_cv_, monitor, filter_);
    infer_worker_ = std::make_shared<InferWorker>(conf_, encode_queue_, encode_mutex_, encode_cv_, infer_queue_, infer_mutex_, infer_cv_,
                                                    pp_queue_, pp_mutex_, pp_cv_, monitor, frontend_id);
    decode_worker_ = std::make_shared<DecodeWorker>(conf_, infer_queue_, infer_mutex_, infer_cv_, nullptr, nullptr, nullptr, monitor, filter_);
    LOG_INFO("Frontend created");
}

Frontend::~Frontend() {}

void Frontend::Exec(const ImageArgs& request){
    LOG_INFO("frontend exec request, filename: %s, model: %s, scale: %s", request.filename.c_str(), request.model_name.c_str(), request.scale.c_str());
    ImageClassifyArgs request_info;
    request_info.id = request.id;
    request_info.scale = request.scale;
    request_info.filename = request.filename;
    request_info.model_name = request.model_name;
    request_info.data.assign(request.data.begin(), request.data.end());
    request_info.stream = request.stream;
    request_info.end_signal_ = request.end_signal_;
    request_info.is_recompute_ = request.is_recompute_;
    request_info.is_parity_data_ = request.is_parity_data_;
    // request_info.frontendId = frontend_id;
    
    std::unique_lock<std::mutex> lock(*recv_mutex_);
    auto query = new SingleQuery(request_info);
    recv_queue_->Push(query);
    lock.unlock();
    LOG_INFO("push query: %d to recv queue", request_info.id);
    recv_cv_->notify_all();
}
