#include "Worker.hh"

std::shared_ptr<std::mutex> Worker::mtx_ = std::make_shared<std::mutex>();
double Worker::cdc_infer_time_ = 0.0;
double Worker::backup_infer_time_ = 0.0;
double Worker::decode_time_ = 0.0;

std::shared_ptr<std::mutex> Worker::mtx_encode_fail_num_= std::make_shared<std::mutex>();
tbb::concurrent_unordered_map<uint64_t, std::pair<uint32_t,uint32_t>> Worker::encode_fail_num;

std::shared_ptr<std::mutex> Worker::mtx_stripes_ = std::make_shared<std::mutex>();
tbb::concurrent_unordered_map<uint64_t, std::unordered_set<uint32_t>>  Worker::stripes; //[encode_id, [id]]

std::shared_ptr<std::mutex> Worker::mtx_backups_ = std::make_shared<std::mutex>();
tbb::concurrent_unordered_map<uint64_t, uint32_t>  Worker::backups; //[encode_id, [id]]

tbb::concurrent_unordered_map<uint64_t, bool>  Worker::is_stripes_completed; //[encode_id, is_completed]

std::mutex replyMtx;
std::condition_variable replyCV;

extern std::shared_ptr<std::mutex> monitorMtx;
extern std::shared_ptr<std::condition_variable> monitorCV;
extern bool is_notify_;

void fastRemoveElement(std::vector<int>& vec, size_t pos) {
    if (pos < vec.size()) {
        std::swap(vec[pos], vec.back());
        vec.pop_back();
    }
}

std::atomic<bool> keep_running(true);

std::vector<std::vector<uint8_t>> generateRandom2DVector(int row, int col) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::vector<std::vector<uint8_t>> result(row, std::vector<uint8_t>(col));
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result[i][j] = static_cast<uint8_t>(dis(gen));
        }
    }
    return result;
}

/**
 * PreprocessWorker
 * 
 */
PreprocessWorker::PreprocessWorker(std::shared_ptr<Config> conf_,
                        std::shared_ptr<QueryQueue> queue_1,
                        std::shared_ptr<std::mutex> mtx_1,
                        std::shared_ptr<std::condition_variable> cv_1,
                        std::shared_ptr<QueryQueue> queue_2,
                        std::shared_ptr<std::mutex> mtx_2,
                        std::shared_ptr<std::condition_variable> cv_2,
                        std::shared_ptr<Monitor> monitor):
                        Worker(conf_, queue_1, mtx_1, cv_1, queue_2, mtx_2, cv_2, monitor)
{
    format = conf_ -> format;
    dtype = conf_ -> dtype;
    channels = conf_ -> channels;
    height = conf_ -> height;
    width = conf_ -> width;

    pp_thread_ = std::thread(&PreprocessWorker::run, this);
}

PreprocessWorker::~PreprocessWorker() {
    pp_thread_.join();
}

void PreprocessWorker::init(){
};

void PreprocessWorker::run() {
    int type1, type3;
    while(true) {
        std::unique_lock<std::mutex> lock(*mtx_1_);
        // LOG_INFO("PreprocessWorker waiting...");
        cv_1_->wait(lock, [this] { return queue_1_->Size() > 0; });
        auto start = std::chrono::high_resolution_clock::now();
        // pop query from recv queue
        // preprocess 
        auto query = dynamic_cast<SingleQuery*>(queue_1_->Pop());

        // judge whether is the end signal
        if(query->end_signal_) {
            std::unique_lock<std::mutex> pp_lock(*mtx_2_);
            queue_2_->Push(query);
            LOG_INFO("push end signal to preprocessed queue");
            cv_2_->notify_all();
            continue;
        }

        int id = query->id_;
        std::string filename = query->filename_;
        std::vector<uint8_t> data = query->data_;
        LOG_INFO("pop query: %d from recv queue", id);
        
        cv::Mat img = imdecode(cv::Mat(data), -1);
        if (img.empty()) {
            LOG_ERROR("error: unable to decode image %s",filename.c_str());
            exit(1);
        }

        std::vector<uint8_t> input_data;

        Preprocessor preprocessor;
        ScaleType scale = preprocessor.ParseScale(query->scale_);
        preprocessor.ParseType(dtype,&type1,&type3);
        preprocessor.Preprocess(img, format, type1, type3, channels, cv::Size(width,height), scale, &input_data);
        query->data_.assign(input_data.begin(),input_data.end());
        
        // push pp query into pp queue
        {
            std::unique_lock<std::mutex> pp_lock(*mtx_2_);
            queue_2_->Push(query);
            LOG_INFO("push query: %d to preprocessed queue", id);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                            (end_time - start).count();
        std::cout << "Preproccessor Worker time: " << duration_time << std::endl;
        cv_2_->notify_all();
    }
}

/**
 * EncodeWorker
 * 
 */
EncodeWorker::EncodeWorker(std::shared_ptr<Config> conf_,
                        std::shared_ptr<QueryQueue> queue_1,
                        std::shared_ptr<std::mutex> mtx_1,
                        std::shared_ptr<std::condition_variable> cv_1,
                        std::shared_ptr<QueryQueue> queue_2,
                        std::shared_ptr<std::mutex> mtx_2,
                        std::shared_ptr<std::condition_variable> cv_2,
                        std::shared_ptr<Monitor> monitor,
                        std::shared_ptr<Filter> filter):
                        Worker(conf_, queue_1, mtx_1, cv_1, queue_2, mtx_2, cv_2,monitor)
{
    encode_thread_ = std::thread(&EncodeWorker::run, this);
    backup_num_ = conf_ -> backup_num;
    k_ = conf_ -> k;
    filter_ = filter;

    if (conf_->encoder_type == "linear") {
        encoder_ = std::make_shared<LinearEncoder>(conf_);

        // LOG_ERROR("Linear Encoder begin to heat!");
        // std::vector<std::vector<uint8_t>> data = generateRandom2DVector(k_, conf_->)
        // for (int i = 0; i < 10; i++) {

        // }
        // LOG_ERROR("Linear Encoder begin to heat!");
    }
    else {
        LOG_ERROR("error encoder type!");
        exit(1);
    }
}

EncodeWorker::~EncodeWorker() {
    encode_thread_.join();
}

void EncodeWorker::init(){

}

void EncodeWorker::run() {
    int encode_id1 = 0 , encode_id2 = 0, data_id = ENCODE_DATA_ID_MIN;
    // std::vector<std::vector<uint8_t>> data;
    std::vector<SingleQuery*> querys{};
    while(true) {
        {
            std::unique_lock<std::mutex> lock(*mtx_1_);
            LOG_INFO("EncodeWorker waiting...");
            cv_1_->wait(lock, [this] { return queue_1_->Size() > 0;});
        }
        auto start = std::chrono::high_resolution_clock::now();

        auto pp_query = dynamic_cast<SingleQuery*>(queue_1_->Pop());
        if(pp_query->end_signal_) { 

            for(auto& query_backup: querys) {
                ImageClassifyArgs request_info = createRequestInfo(*query_backup, "Backup", encode_id2,
                                                                     true, false, query_backup->is_recompute_);
                request_info.data.assign(query_backup->data_.begin(), query_backup->data_.end());

                {
                    query_backup->encode_id_ = encode_id2;
                    query_backup->encode_type_ = "Backup";
                    backups[encode_id2] = query_backup->id_;
                    std::unique_lock<std::mutex> encode_lock(*mtx_2_);
                    queue_2_->Push(query_backup);
                    LOG_INFO("push query: %d to encode queue, encode id: %d", query_backup->id_, encode_id2);
                }

                for(int i = 0; i < backup_num_; i++) {
                    auto query = new SingleQuery(request_info);
                    query -> id_ = data_id++;
                    query -> filename_ = "backup_" + std::to_string(i) + "_" + std::to_string(encode_id2);
                    std::unique_lock<std::mutex> encode_lock(*mtx_2_);
                    queue_2_->Push(query);
                    LOG_INFO("push query: %d to encode queue, encode id: %d", query->id_, encode_id2);
                    encode_lock.unlock();
                }
                encode_id2++;
            }
            std::vector<SingleQuery*>().swap(querys);

            std::unique_lock<std::mutex> encode_lock(*mtx_2_);
            queue_2_->Push(pp_query);
            LOG_INFO("push push end signal encode queue");
            encode_lock.unlock();  
            cv_2_->notify_all(); 
            continue;
        }

        assert(pp_query->id_ < ENCODE_DATA_ID_MIN);
        LOG_INFO("pop query: %d from pp queue", pp_query->id_);

        // choose the CDC or backup
        EncodeType encodeType;
        filter_->filterWorker(encodeType);
        // CDC 
        if(encodeType == EncodeType::CDC && !pp_query->is_recompute_) {
            querys.emplace_back(pp_query);
            // {
            //     pp_query->encode_id_ = encode_id1;
            //     pp_query->encode_type_ = "CDC";
            //     std::unique_lock<std::mutex> encode_lock(*mtx_2_);
            //     queue_2_->Push(pp_query);
            // }
            // LOG_INFO("push query: %d to encoded queue, CDC encode id: %d", pp_query->id_, encode_id1);

            if(querys.size() == k_) {
                ImageClassifyArgs request_info = createRequestInfo(*pp_query, "CDC", encode_id1, true, false, pp_query->is_recompute_);
                std::vector<std::vector<uint8_t>> data{};
                for(auto& query: querys) {
                    data.emplace_back(query->data_);
                    std::unique_lock<std::mutex> encode_lock(*mtx_2_);
                    query->encode_id_ = encode_id1;
                    query->encode_type_ = "CDC";
                    queue_2_->Push(query);
                    LOG_INFO("push query: %d to encode queue, CDC encode id: %d", query->id_, encode_id1);
                    encode_lock.unlock(); 
                    stripes[encode_id1].insert(query->id_);
                }
                
                encoder_->encode(data, request_info.data);
                auto query = new SingleQuery(request_info);
                query->id_ = data_id++;
                query->filename_ = "encode_"+ std::to_string(encode_id1);
                LOG_INFO("Generate query: %d, data size: %ld", query->id_, query->data_.size());

                std::unique_lock<std::mutex> encode_lock(*mtx_2_);
                queue_2_->Push(query);
                LOG_INFO("push query: %d to encode queue, CDC encode id: %d", query->id_, encode_id1);
                encode_lock.unlock();
                stripes[encode_id1].insert(query->id_);  

                is_stripes_completed[encode_id1] = false;

                encode_id1++;
                std::vector<SingleQuery*>().swap(querys);
            }
        }
        // Backup
        else if(encodeType == EncodeType::Backup || pp_query->is_recompute_) {

            // data.emplace_back(pp_query->data_);
            ImageClassifyArgs request_info = createRequestInfo(*pp_query, "Backup", encode_id2, true, false, pp_query->is_recompute_);
            request_info.data.assign(pp_query->data_.begin(), pp_query->data_.end());

            {
                pp_query->encode_id_ = encode_id2;
                pp_query->encode_type_ = "Backup";
                std::unique_lock<std::mutex> encode_lock(*mtx_2_);
                backups[encode_id2] = pp_query->id_;
                queue_2_->Push(pp_query);
                LOG_INFO("push query: %d to encode queue, encode id: %d", pp_query->id_, encode_id2);
            }

            for(int i = 0; i < backup_num_; i++) {
                // 
                // ReplicaEncoder encoder(conf_);
                // encoder.encode(data, request_info.data);
                auto query = new SingleQuery(request_info);
                query -> id_ = data_id++;
                query -> filename_ = "backup_" + std::to_string(i) + "_" + std::to_string(encode_id2);
                std::unique_lock<std::mutex> encode_lock(*mtx_2_);
                queue_2_->Push(query);
                LOG_INFO("push query: %d to encode queue, encode id: %d", query->id_, encode_id2);
                encode_lock.unlock();
            }
            // std::vector<std::vector<uint8_t>>().swap(data);
            encode_id2++;    
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                            (end_time - start).count();
        std::cout << "Encoder Worker time: " << duration_time << std::endl;
        
        cv_2_->notify_all(); 
    }
}

ImageClassifyArgs EncodeWorker::createRequestInfo(const SingleQuery& query,
                                                    const std::string& encodeType,
                                                    int encode_id,
                                                    bool is_parity_data,
                                                    bool is_end_signal,
                                                    bool is_recompute) {  
    ImageClassifyArgs request_info;
    request_info.model_name = query.model_name_;
    request_info.scale = query.scale_;
    request_info.filename = query.filename_;
    request_info.stream = query.stream_;
    request_info.id = query.id_;
    request_info.encode_type = encodeType;
    request_info.encode_id = encode_id;
    request_info.is_parity_data_ = is_parity_data;
    request_info.end_signal_ = is_end_signal;
    request_info.is_recompute_ = is_recompute;
    return request_info;
}

/**
 *InferWorker
 * 
 */
InferWorker::InferWorker(std::shared_ptr<Config> conf_,
                        std::shared_ptr<QueryQueue> queue_1,
                        std::shared_ptr<std::mutex> mtx_1,
                        std::shared_ptr<std::condition_variable> cv_1,
                        std::shared_ptr<QueryQueue> queue_2,
                        std::shared_ptr<std::mutex> mtx_2,
                        std::shared_ptr<std::condition_variable> cv_2,
                        std::shared_ptr<QueryQueue> queue_3,
                        std::shared_ptr<std::mutex> mtx_3,
                        std::shared_ptr<std::condition_variable> cv_3,
                        std::shared_ptr<Monitor> monitor,
                        std::uint32_t frontend_id):
                        Worker(conf_, queue_1, mtx_1, cv_1, queue_2, mtx_2, cv_2, monitor)
{
    node_number_ = conf_ -> node_number;
    frontend_id_ = frontend_id;
    queue_3_ = queue_3;
    mtx_3_ = mtx_3;
    cv_3_ = cv_3;

    mtx_map_ = std::make_shared<std::mutex>();
    mtx_start_time_map_ = std::make_shared<std::mutex>();
    mtx_querys_encode_map_ = std::make_shared<std::mutex>();
    mtx_querys_backup_map_ = std::make_shared<std::mutex>();
    mtx_backup_fail_num_ = std::make_shared<std::mutex>();
    mtx_visited_id1_ = std::make_shared<std::mutex>();
    mtx_visited_id2_ = std::make_shared<std::mutex>();

    // std::stringstream ss(conf_->model_name);
    // std::getline(ss, model_name_, '-');
    model_name_ = conf_->model_name;

    visited_id1_ = std::make_shared<QueryIdSet>();
    visited_id2_ = std::make_shared<QueryIdSet>();

    infer_thread_ = std::thread(&InferWorker::run, this);
}

InferWorker::~InferWorker() {
    for (auto& thread : recv_threads_) {
        thread.join();
    }

    infer_thread_.join();
    querys_map_.clear();
    querys_start_time_map_.clear();
}

void InferWorker::init(){

}

std::shared_ptr<std::mutex> get_mutex_for_region(std::unordered_map<std::uint32_t, std::shared_ptr<std::mutex>> chosenRegionMutex, std::uint32_t encode_id) {
    static std::mutex global_mutex;
    std::lock_guard<std::mutex> lock(global_mutex);

    auto& mtx_ptr = chosenRegionMutex[encode_id];
    if (!mtx_ptr) {
        mtx_ptr = std::make_shared<std::mutex>();
    }
    return mtx_ptr;
}

void InferWorker::run() {
    // std::vector<std::shared_ptr<grpcStreamClient>> streams(node_number_,nullptr);  
    // std::vector<std::shared_ptr<ClientContext>> contexts(node_number_,nullptr);
    std::unordered_map<std::string, std::shared_ptr<grpcStreamClient>> streams;
    std::unordered_map<std::string, std::shared_ptr<ClientContext>> contexts;
    std::unordered_map<std::uint32_t, std::unordered_set<std::string>> CDCchosenIP;
    std::unordered_map<std::uint32_t, std::unordered_set<std::string>> BackupchosenIP;
    std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>> chosenRegion;
    std::unordered_map<std::uint32_t, std::shared_ptr<std::mutex>> chosenRegionMutex;

    while(true) {
        {
            std::unique_lock<std::mutex> lock(*mtx_1_);
            LOG_INFO("InferWorker waiting...");
            cv_1_->wait(lock, [this] { return queue_1_->Size() > 0; });
        }

        auto start = std::chrono::high_resolution_clock::now();
    
        auto encode_query = dynamic_cast<SingleQuery*>(queue_1_->Pop());
        if(encode_query->end_signal_) {
            for(const auto& stream: streams){
                ElasticcdcRequest request;
                request.set_end_signal(true);
                request.set_id(encode_query->id_);
                request.set_frontend_id(frontend_id_);
                request.set_recompute(encode_query->is_recompute_);
                stream.second->Write(request);
                LOG_INFO("Client sending end sigal to backend %s", stream.first.c_str());
            }
            continue;
        }

        if (encode_query->encode_type_ == "CDC") {
            if (CDCchosenIP.find(encode_query->encode_id_) == CDCchosenIP.end()) {
                std::unordered_set<std::string> emptySet;
                std::unordered_set<std::uint32_t> emptySet2;
                CDCchosenIP[encode_query->encode_id_] = emptySet;
            }
        }
        else if (encode_query->encode_type_ == "Backup") {
            if (BackupchosenIP.find(encode_query->encode_id_) == BackupchosenIP.end()) {
                std::unordered_set<std::string> emptySet;
                std::unordered_set<std::uint32_t> emptySet2;
                BackupchosenIP[encode_query->encode_id_] = emptySet;
                chosenRegion[encode_query->encode_id_] = emptySet2;
            }
        }
        
            

        LOG_INFO("pop query: %d from encode queue", encode_query->id_);
        
        // // choose a backend to send query randomly
        // std::srand(static_cast<unsigned int>(std::time(nullptr)));
        // int node_index = std::rand() % node_number_;
        // LOG_INFO("Choose backend id: %d to send data", node_index);

        // choose a backend to send query randomly

        std::vector<std::string> backend_ip_list{};
        if(monitor_->findAvaBackendIPs()) {
            if(encode_query->encode_type_ == "CDC") {
                auto ava_invul_backendIPS = monitor_->get_ava_invul_backendIPS();
                auto ava_vul_backendIPS = monitor_->get_ava_vul_backendIPS();
                if(encode_query->is_parity_data_) {
                    backend_ip_list = ava_vul_backendIPS.empty() ? ava_invul_backendIPS : ava_vul_backendIPS;
                } else {
                    backend_ip_list = ava_invul_backendIPS.empty() ? ava_vul_backendIPS : ava_invul_backendIPS;
                }
                // std::cout << "Available invulnerable nodes size:" << ava_invul_backendIPS.size() << std::endl;
                // for(const auto& backend_ip: ava_invul_backendIPS) {
                //     std::cout << backend_ip << std::endl;
                // }
    
                // std::cout << "Available vulnerable nodes size:" << ava_vul_backendIPS.size() << std::endl;
                // for(const auto& backend_ip: ava_vul_backendIPS) {
                //     std::cout << backend_ip << std::endl;
                // }
            } else {
                backend_ip_list = monitor_->get_ava_backendIPs();
            }
           
            // std::cout << "Choose nodes size:" << backend_ip_list.size() << std::endl;
            // for(const auto& backend_ip: backend_ip_list) {
            //     std::cout << backend_ip << std::endl;
            // }
        }
        else {
            LOG_INFO("Choose on-demand node!");
            backend_ip_list = monitor_->get_all_backendIPs();
        }

        std::string backendIP = ""; 
        if (encode_query->encode_type_ == "CDC") {
            auto is_parity_data = encode_query->is_parity_data_;
            do {
                // backendIP = monitor_->ChooseABackend(is_parity_data);
                auto it = std::find(backend_ip_list.begin(), backend_ip_list.end(), backendIP);
                if (it != backend_ip_list.end()) {
                    backend_ip_list.erase(it);
                }
                if (backend_ip_list.size() == 0) {
                    backend_ip_list = encode_query->is_parity_data_ ? monitor_->get_ava_invul_backendIPS(): monitor_->get_ava_vul_backendIPS();
                }
                assert(backend_ip_list.size() > 0);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, backend_ip_list.size() - 1);
                backendIP = backend_ip_list[dis(gen)];
            } while(CDCchosenIP[encode_query->encode_id_].find(backendIP) != CDCchosenIP[encode_query->encode_id_].end());
            CDCchosenIP[encode_query->encode_id_].insert(backendIP);
        }
        else if (encode_query->encode_type_ == "Backup") {
            auto mtx = get_mutex_for_region(chosenRegionMutex, encode_query->id_);
            std::lock_guard<std::mutex> lock(*mtx);
            std::uint32_t regionid;
            if (monitor_->avaRegionNum() > 1) {
                do {
                    auto it = std::find(backend_ip_list.begin(), backend_ip_list.end(), backendIP);
                    if (it != backend_ip_list.end()) {
                        backend_ip_list.erase(it);
                    }
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_int_distribution<> dis(0, backend_ip_list.size() - 1);
                    backendIP = backend_ip_list[dis(gen)];
                    regionid = monitor_->Ip2Region(backendIP);
                } while(BackupchosenIP[encode_query->encode_id_].count(backendIP) > 0
                        || chosenRegion[encode_query->encode_id_].count(regionid) > 0);
            } 
            else {
                do {
                    auto it = std::find(backend_ip_list.begin(), backend_ip_list.end(), backendIP);
                    if (it != backend_ip_list.end()) {
                        backend_ip_list.erase(it);
                    }
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_int_distribution<> dis(0, backend_ip_list.size() - 1);
                    backendIP = backend_ip_list[dis(gen)];
                    regionid = monitor_->Ip2Region(backendIP);
                } while(BackupchosenIP[encode_query->encode_id_].count(backendIP) > 0);
            }
            chosenRegion[encode_query->encode_id_].insert(regionid);
            BackupchosenIP[encode_query->encode_id_].insert(backendIP);
        }
        else {
            LOG_INFO("error encode type!");
            exit(-1);
        }

        monitor_->sendQueryToABackend(backendIP, encode_query->id_, encode_query->encode_id_, encode_query->encode_type_ == "CDC");

        // if (encode_query->is_recompute_) {
        //     LOG_INFO("Recompute: %d, Choose backend IP: %s to send data", encode_query->encode_id_, backendIP.c_str());
        // }
        // else{
        LOG_INFO("Encode id: %d, Choose backend IP: %s to send data", encode_query->encode_id_, backendIP.c_str());
        // }

        // if(!streams[node_index]) {
        if(streams.find(backendIP) == streams.end()) {
            // std::string backendIP = conf_ -> backend_IPs[node_index];
            // LOG_INFO("choose backend server: %s", backendIP.c_str());
            std::string backendIPProt = backendIP + ":" + "50051";
            std::shared_ptr<Channel> channel = grpc::CreateChannel(
                backendIPProt, grpc::InsecureChannelCredentials());

            LOG_INFO("Create channel %s", backendIPProt.c_str());
            std::unique_ptr<ElasticcdcService::Stub> stub = ElasticcdcService::NewStub(channel);
            // LOG_INFO("Create stub sucessfully");

            std::shared_ptr<ClientContext> context = std::make_unique<ClientContext>();
            
            std::shared_ptr<grpcStreamClient> stream(
                stub->DataTransStream(context.get()));
            // LOG_INFO("Create stream sucessfully");
            
            // streams[node_index] = stream;
            // contexts[node_index] = context;
            streams[backendIP] = stream;
            contexts[backendIP] = context;

            std::thread recv_thread_ = std::thread(&InferWorker::ReceiveResponses, this, stream, backendIP);
            // LOG_INFO("Create recv thread sucessfully");
            recv_threads_.emplace_back(std::move(recv_thread_));
        }

        ElasticcdcRequest request;
        // fill the request info
        assert(encode_query!=nullptr);
        assert(!encode_query->data_.empty());
        request.set_data(reinterpret_cast<const char*>
                            (encode_query->data_.data()), encode_query->data_.size());
        request.set_model_name(encode_query->model_name_);
        request.set_scale(encode_query->scale_);
        request.set_filename(encode_query->filename_);
        request.set_id(encode_query->id_);
        request.set_encode_type(encode_query->encode_type_);
        request.set_frontend_id(frontend_id_);
        request.set_end_signal(false);
        request.set_recompute(encode_query->is_recompute_);
        // std::cout << "encode_query->encode_type_:" << encode_query->encode_type_ << std::endl;
        {
            std::unique_lock<std::mutex> lock(*mtx_);
            request.set_cdc_infer_time(cdc_infer_time_);
            request.set_backup_infer_time(backup_infer_time_);
            request.set_decode_time(decode_time_);
            std::cout << "cdc_infer_time:" << cdc_infer_time_
                         << " backup_infer_time:" << backup_infer_time_
                         << " decode_time:" << decode_time_ << std::endl;
            
        }
        
        // assert(streams[node_index] != nullptr);
        // streams[node_index]->Write(request);
        querys_map_[encode_query->id_] = encode_query;
        querys_start_time_map_[encode_query->id_] = std::chrono::high_resolution_clock::now();
        assert(streams[backendIP] != nullptr);
        streams[backendIP]->Write(request);

        // notify monitor
        if (!(encode_query->is_parity_data_ || encode_query->is_recompute_)) {
            query_num++;
            // std::cout << "query_num:" << query_num << " conf -> update_interval:" << conf_ -> update_interval << std::endl;
            if(conf_->update_mode == "query" && ! (query_num % conf_ -> update_interval)) {
                // std::cout << "query_num:" << query_num << std::endl;
                std::lock_guard<std::mutex> lock(*monitorMtx);
                is_notify_ = true;
                monitorCV->notify_one();
            }
        } 

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                            (end_time - start).count();
        std::cout << "Infer Worker send time: " << duration_time << std::endl;

        LOG_INFO("Client sending image %s with size: %ld bytes, is recompute:%d ", 
                    encode_query->filename_.c_str(), encode_query->data_.size(), encode_query->is_recompute_);
    }
}

void InferWorker::ReceiveResponses(std::shared_ptr<grpcStreamClient> stream, std::string backend_ip){
    LOG_INFO("Wait for responses from the backend");
    ElasticcdcReply reply;
   
    while (stream->Read(&reply)) {
        auto start = std::chrono::high_resolution_clock::now();
        
        int id = reply.id();
        auto recv_query = dynamic_cast<SingleQuery*>(querys_map_[id]);
        LOG_INFO("Receive the image %d from the backend, is parity data %d", id, recv_query->is_parity_data_);
        
        assert(querys_start_time_map_.find(id) != nullptr);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                                            (end - querys_start_time_map_[id]).count();
        recv_query -> infer_time = duration;
        // need to rethink
        {
            std::lock_guard<std::mutex> lock(*mtx_start_time_map_);
            querys_start_time_map_.unsafe_erase(id);
        }
        

        assert(querys_map_.find(id) != nullptr); 
        {
            std::lock_guard<std::mutex> lock(*mtx_map_);
            querys_map_.unsafe_erase(id);
        }

       
        // judge the backend is fail or not, and decide if need to recalculate, before this, judge whether is the on-demand nodes
        // bool is_preempted = false;
        // if(monitor_->findAvaBackendIPs()) {
        //     auto ava_backend_ips = monitor_->get_ava_backendIPs();
        //     is_preempted = std::find(ava_backend_ips.begin(), ava_backend_ips.end(), backend_ip) == ava_backend_ips.end();
        // }
        bool is_preempted = monitor_->IsQueryBroken(backend_ip, id);
        monitor_->DeleteAQueryState(backend_ip, id);
        
        if(recv_query->encode_type_ == "CDC") {
            encode_fail_num[recv_query->encode_id_].second++;
            if(is_preempted){
                encode_fail_num[recv_query->encode_id_].first++;
                querys_encode_map_[recv_query->encode_id_].emplace_back(recv_query);
            }
            if(encode_fail_num[recv_query->encode_id_].second == conf_->k + 1
                && encode_fail_num[recv_query->encode_id_].first > 1) {
                for(auto& recalc_query : querys_encode_map_[recv_query->encode_id_]) {
                    if(recalc_query->is_parity_data_) continue;
                    std::unique_lock<std::mutex> recv_lock(*mtx_3_);
                    recalc_query->is_recompute_ = true;
                    queue_3_->Push(recalc_query);
                    LOG_INFO("CDC query %d recompute!", recalc_query->id_);
                    LOG_INFO("push query: %d to recv queue", recalc_query->id_);
                    recv_lock.unlock();  
                    cv_3_->notify_all();
                }
                
                {
                    std::lock_guard<std::mutex> lock(*mtx_querys_encode_map_);
                    querys_encode_map_.unsafe_erase(recv_query->encode_id_);
                }
                
            }

        } else if(recv_query->encode_type_ == "Backup") {
            std::cout << "query " << recv_query->id_ << " is preempted " << is_preempted << ", encode id " << recv_query->encode_id_ << " fail num: " << backup_fail_num[recv_query->encode_id_] << std::endl;
            if(is_preempted){
                backup_fail_num[recv_query->encode_id_]++;
                if(!recv_query -> is_parity_data_)
                    querys_backup_map_[recv_query->encode_id_].emplace_back(recv_query);
            }

            if(backup_fail_num[recv_query->encode_id_] == 1 + conf_->backup_num) {
                auto recalc_query = querys_backup_map_[recv_query->encode_id_][0];
                std::unique_lock<std::mutex> recv_lock(*mtx_3_);
                recalc_query->is_recompute_ = true;
                queue_3_->Push(recalc_query);
                LOG_INFO("Backup recompute!");
                LOG_INFO("push query: %d to recv queue", recalc_query->id_);
                recv_lock.unlock();  
                cv_3_->notify_all();
                {
                    std::lock_guard<std::mutex> lock(*mtx_backup_fail_num_);
                    backup_fail_num.unsafe_erase(recv_query->encode_id_);
                }
            
                {
                    std::lock_guard<std::mutex> lock(*mtx_querys_backup_map_);
                    querys_backup_map_.unsafe_erase(recv_query->encode_id_);
                }
                
            }
        }
        

        if(is_preempted) {

            continue;
        }

        // update the infer_time 
        {
            std::unique_lock<std::mutex> lock(*mtx_);
            if(recv_query -> encode_type_ == "CDC") 
                cdc_infer_time_ = duration;
            else backup_infer_time_ = duration;

            LOG_INFO("Update infer time, cdc: %lf, backup: %lf", cdc_infer_time_, backup_infer_time_);
        }
        
        assert(recv_query!=nullptr);
        assert(!reply.reply_info().empty());
        // std::cout << "reply.reply_info():" << reply.reply_info() << std::endl;
        recv_query->reply_info_bytes.assign(reply.reply_info().begin(), reply.reply_info().end());


        SendToClient(recv_query);

        
        // std::unique_lock<std::mutex> infer_lock(*mtx_2_);
        // queue_2_->Push(recv_query);
        // LOG_INFO("push query: %d to infered queue", recv_query->id_);
        // infer_lock.unlock();  

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                            (end_time - start).count();
        std::cout << "Backend " << backend_ip << " Infer Worker receive time: " << duration_time << std::endl;
             
    }

    Status status = stream->Finish();
    if (status.ok()) {
        LOG_INFO("RPC completed successfully");
    } else {
        LOG_ERROR("RPC failed: %s",status.error_message().c_str());
    }

    // std::unique_lock<std::mutex> infer_lock(*mtx_2_);
    // queue_2_->Push(encode_query);
    // LOG_INFO("push query: %d to infered queue", encode_query->id_);
    // infer_lock.unlock();  
    // cv_2_->notify_all();
}


// void InferWorker::SendToClient(SingleQuery* recv_query) {
//     std::cout << "SendToClient" << std::endl;
//     // send the original query to the client (CDC)
//     auto encode_id = recv_query -> encode_id_;
//     if(recv_query -> encode_type_ == "CDC") {
//         std::cout << 0 << std::endl;
//         {
//             std::unique_lock<std::mutex> task_lock(*mtx_visited_id1_);
//             if(visited_id1_->contains(encode_id))  return;
//             if(received_querys_num_[encode_id] == conf_ -> k - 1) {
//                 visited_id1_->insert(encode_id);
//             } 
//         }
//         std::cout << "encode_id:" << encode_id << std::endl;

//         if(++received_querys_num_[encode_id] <= conf_ -> k)   {
//             std::cout << 3 << std::endl;
//             if(!recv_query->is_parity_data_) {
//                 ElasticcdcReply reply;
//                 std::string reply_data(recv_query->reply_info_bytes.begin()+ DATASETS.at(model_name_).first,
//                                     recv_query->reply_info_bytes.end());
//                 reply.set_reply_info(reply_data);
//                 reply.set_id(recv_query->id_ - start_task_id);
//                 reply.set_recompute(recv_query->is_recompute_);
//                 std::cout << 4 << std::endl;
//                 if (recv_query->stream_) {
//                     LOG_INFO("Send query %d to client as %d.", recv_query->id_, recv_query->id_ - start_task_id);
                    
//                     std::unique_lock<std::mutex> task_lock(*taskCountMtx);
//                     recv_query->stream_->Write(reply);
//                     tasks_completed_num++;
//                 }
//             } 
//             std::unique_lock<std::mutex> infer_lock(*mtx_2_);
//             queue_2_->Push(recv_query);
//             LOG_INFO("push query: %d to infered queue", recv_query->id_);
//             infer_lock.unlock(); 
//             cv_2_->notify_all(); 
//         }

          
//     } else if (recv_query -> encode_type_ == "Backup") {
//         std::cout << 5 << std::endl;
//         {
//             std::unique_lock<std::mutex> task_lock(*mtx_visited_id2_);
//             if(visited_id2_->contains(encode_id)) return;
//             visited_id2_->insert(encode_id);
//         }
//         std::cout << "encode_id:" << encode_id << std::endl;
//         auto backup_id = backups[encode_id];
//         std::cout << "backup_id:" << backup_id << std::endl;
//         ElasticcdcReply reply;
//         std::string reply_data(recv_query->reply_info_bytes.begin() + DATASETS.at(model_name_).first,
//                                 recv_query->reply_info_bytes.end());
//         std::cout << "1" << backup_id << std::endl;
//         reply.set_reply_info(reply_data);
//         reply.set_id(backup_id - start_task_id);
//         reply.set_recompute(recv_query->is_recompute_);
//         std::cout << "2" << backup_id << std::endl;
//         if (recv_query->stream_) {
//             LOG_INFO("Send query %d to client as %d.", backup_id, backup_id - start_task_id);
//             std::unique_lock<std::mutex> task_lock(*taskCountMtx);
//             recv_query->stream_->Write(reply);
//             tasks_completed_num++;
//         }
        
//         {
//             std::unique_lock<std::mutex> lock(*mtx_backups_);
//             backups.unsafe_erase(encode_id);
//         }
        
//     }

//     {
//         std::unique_lock<std::mutex> reply_lock(completedMutex);
//         completedCV.notify_all();
//     }
// }

void InferWorker::SendToClient(SingleQuery* recv_query) {
    // std::cout << "SendToClient" << std::endl;
    // send the original query to the client (CDC)
    auto encode_id = recv_query -> encode_id_;
    if(recv_query -> encode_type_ == "CDC") {

        if(is_stripes_completed[encode_id]) return;

        // std::cout << "encode_id:" << encode_id << std::endl;
        if(!recv_query->is_parity_data_) {
            ElasticcdcReply reply;
            std::vector<std::string> reply_data_vec;
            if (conf_->model_name.find("irevnet") != std::string::npos) {
                std::string reply_data(recv_query->reply_info_bytes.begin() + DATASETS.at(model_name_).first,
                                    recv_query->reply_info_bytes.end());
                reply_data_vec.push_back(reply_data);
            }
            else {
                std::string reply_data(recv_query->reply_info_bytes.begin(),
                                    recv_query->reply_info_bytes.end());
                reply_data_vec.push_back(reply_data);
            }
            
            reply.set_reply_info(reply_data_vec[0]);
            reply.set_id(recv_query->id_ - start_task_id);
            reply.set_recompute(recv_query->is_recompute_);
            if (recv_query->stream_) {
                LOG_INFO("Send query %d to client as %d.", recv_query->id_, recv_query->id_ - start_task_id);
                    
                std::unique_lock<std::mutex> task_lock(*taskCountMtx);
                if(!is_stripes_completed[encode_id]) {
                    recv_query->stream_->Write(reply);
                    tasks_completed_num++;
                    received_querys_num_[encode_id]++;
                    if(received_querys_num_[encode_id] == conf_->k) is_stripes_completed[encode_id] = true;
                }
            }
        } 
        std::unique_lock<std::mutex> infer_lock(*mtx_2_);
        queue_2_->Push(recv_query);
        LOG_INFO("push query: %d to infered queue", recv_query->id_);
        infer_lock.unlock(); 
        cv_2_->notify_all(); 
          
    } else if (recv_query -> encode_type_ == "Backup") {
        {
            std::unique_lock<std::mutex> task_lock(*mtx_visited_id2_);
            if(visited_id2_->contains(encode_id)) return;
            visited_id2_->insert(encode_id);
        }
        std::cout << "encode_id:" << encode_id << std::endl;
        auto backup_id = backups[encode_id];
        std::cout << "backup_id:" << backup_id << std::endl;
        ElasticcdcReply reply;
        std::vector<std::string> reply_data_vec;
        if (conf_->model_name.find("irevnet") != std::string::npos) {
            std::string reply_data(recv_query->reply_info_bytes.begin() + DATASETS.at(model_name_).first,
                                recv_query->reply_info_bytes.end());
            reply_data_vec.push_back(reply_data);
        }
        else {
            std::string reply_data(recv_query->reply_info_bytes.begin(),
                                recv_query->reply_info_bytes.end());
            reply_data_vec.push_back(reply_data);
        }
        reply.set_reply_info(reply_data_vec[0]);
        reply.set_id(backup_id - start_task_id);
        reply.set_recompute(recv_query->is_recompute_);
        if (recv_query->stream_) {
            LOG_INFO("Send query %d to client as %d.", backup_id, backup_id - start_task_id);
            std::unique_lock<std::mutex> task_lock(*taskCountMtx);
            recv_query->stream_->Write(reply);
            tasks_completed_num++;
        }
        {
            std::unique_lock<std::mutex> lock(*mtx_backups_);
            backups.unsafe_erase(encode_id);
        }
    }

    {
        std::unique_lock<std::mutex> reply_lock(completedMutex);
        completedCV.notify_all();
    }
}

/**
 * DecodeWorker
 * 
 */
DecodeWorker::DecodeWorker(std::shared_ptr<Config> conf_,
                        std::shared_ptr<QueryQueue> queue_1,
                        std::shared_ptr<std::mutex> mtx_1,
                        std::shared_ptr<std::condition_variable> cv_1,
                        std::shared_ptr<QueryQueue> queue_2,
                        std::shared_ptr<std::mutex> mtx_2,
                        std::shared_ptr<std::condition_variable> cv_2,
                        std::shared_ptr<Monitor> monitor,
                        std::shared_ptr<Filter> filter):
                        Worker(conf_, queue_1, mtx_1, cv_1, queue_2, mtx_2, cv_2, monitor)
{
    k_ = conf_ -> k;
    filter_ = filter;

    // std::stringstream ss(conf_->model_name);
    // std::getline(ss, model_name_, '-');
    model_name_ = conf_->model_name;

    if (conf_->decoder_type == "linear") {
        decoder_ = std::make_shared<LinearDecoder>(conf_);
    }
    else if (conf_->decoder_type == "distill") {
        decoder_ = std::make_shared<DistilledDecoder>(conf_);

        // warmup
        LOG_INFO("Distilled decoder begins to warmup!");
        
        for (int i = 0; i < 5; i++) {
            // auto start = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> res;
            std::vector<uint32_t> shape = DATASETS.at(conf_->model_name).second;
            size_t size = 4;
            for (auto dim: shape) 
                size *= dim;
            std::vector<std::vector<uint8_t>> data = generateRandom2DVector(k_, size);
            decoder_->decode(data, res);
            // auto end = std::chrono::high_resolution_clock::now();
            // auto duration_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
            //                 (end - start).count();

            // std::cout << "Warmup time: " << duration_time << std::endl;
        }
        LOG_INFO("Distilled decoder finishes to warmup!");

        // background_thread_ = std::thread([&]() mutable {
        //     while (keep_running) {
        //         try {
        //             std::vector<std::vector<uint8_t>> data = generateRandom2DVector(k_, DATASETS.at(model_name_).first);
        //             std::vector<uint8_t> res;
        //             decoder_->decode(data, res);
        //         } catch (const std::exception& e) {
        //             std::cerr << "[Dummy Inference Error] " << e.what() << std::endl;
        //         }

        //         std::this_thread::sleep_for(std::chrono::seconds(1)); 
        //     }
        // });

        decode_thread_ = std::thread(&DecodeWorker::run, this);
    }
    else {
        LOG_ERROR("error decoder type!");
        exit(1);
    }
}

DecodeWorker::~DecodeWorker() {
    decode_thread_.join();
    keep_running = false;
    background_thread_.join();
}

void DecodeWorker::init() {

}
void DecodeWorker::run() {
    // int dncode_id1 = 0, dncode_id2 = 0;
    std::unordered_map<int, std::vector<Query*> > querys_id1;   //[decode_id, Querys]
    std::unordered_set<int> visited_id1;    
    std::unordered_set<int> visited_id2;

    std::unordered_map<int, std::chrono::high_resolution_clock::time_point> decode_start_time;  //[decode_id, time]
    std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>> decode_start_time1;  //[decode_id, time]
    
    std::vector<uint8_t> res;
    std::vector<std::vector<uint8_t>> data = generateRandom2DVector(k_, DATASETS.at(conf_->model_name).first);
    decoder_->decode(data, res);
    
    while(true) {
        {
            std::unique_lock<std::mutex> lock(*mtx_1_);
            LOG_INFO("DecodeWorker waiting...");
            cv_1_->wait(lock, [this] { return queue_1_->Size() > 0; });
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto infer_query = dynamic_cast<SingleQuery*>(queue_1_->Pop());
        LOG_INFO("pop query: %d from infer queue, decode id: %d, decode type %s",
                     infer_query->id_, infer_query->encode_id_, infer_query->encode_type_.c_str());

        // disordered
        if(infer_query->encode_type_ == "CDC") {
            
            if(is_stripes_completed[infer_query->encode_id_]) continue;

            if(conf_->test_mode == "normal") {
                // if(conf_->flag_algorithm == "passive") {
                    handleCDCQuery(querys_id1, visited_id1, infer_query, decode_start_time, decode_start_time1);
                    handleLastQueryBroken(querys_id1, decode_start_time);
                // } else if(conf_->flag_algorithm == "baseline") {
                    // handleCDCQueryFlagAlgBaseline(querys_id1, visited_id1, infer_query, decode_start_time, decode_start_time1);
                // }
                
            }
                
            else if(conf_->test_mode == "baseline") 
                handleCDCQueryBaseline(querys_id1, visited_id1, infer_query, decode_start_time, decode_start_time1);
            else {
                LOG_ERROR("Test mode error!");
            }
        }
        // else if(infer_query->encode_type_ == "Backup") {
        //     handleBackupQuery(visited_id2, infer_query);
        // }
        else {
            LOG_ERROR("Encode type error: %s", infer_query->encode_type_.c_str());
        }

        // deal with the encode_broken querys
        handleBrokenQuery(querys_id1, visited_id1, infer_query, decode_start_time);
        

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                            (end_time - start).count();

        std::cout << "Decode Worker time: " << duration_time << std::endl;

        {
            std::unique_lock<std::mutex> reply_lock(completedMutex);
            completedCV.notify_all();
        }
    }
}


/**
 * handle CDC reply
 */
void DecodeWorker::handleCDCQuery(std::unordered_map<int, std::vector<Query*>>& querys_id1,
                                    std::unordered_set<int>& visited_id1,
                                    SingleQuery* infer_query,
                                    std::unordered_map<int, 
                                     std::chrono::high_resolution_clock::time_point>& decode_start_time,
                                    std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>>& decode_start_time1) {
    LOG_INFO("handle CDC query!");
    int decode_id = infer_query->encode_id_;
    // auto visited1_it = visited_id1.find(decode_id);
    
    // the decoding task of decode_id is not over
    // if (visited1_it == visited_id1.end()) {
        auto querys_it = querys_id1.find(decode_id);
        // first receive the query of the decode_id
        if (querys_it == querys_id1.end()) {
            querys_id1[decode_id] = {infer_query};
            // if(!infer_query->is_parity_data_) {
            //     ElasticcdcReply reply;
            //     std::string reply_data(infer_query->reply_info_bytes.begin()+ DATASETS.at(model_name_).first,
            //                                     infer_query->reply_info_bytes.end());
            //     reply.set_reply_info(reply_data);
            //     reply.set_id(infer_query->id_ - start_task_id);
            //     reply.set_recompute(infer_query->is_recompute_);
            //     if (infer_query->stream_) {
            //         LOG_INFO("Send query %d to client as %d.", infer_query->id_, infer_query->id_ - start_task_id);
            //         infer_query->stream_->Write(reply);
            //         tasks_completed_num++;
            //     }
            // }
            decode_start_time[decode_id] = std::chrono::high_resolution_clock::now();
            decode_start_time1[decode_id] = std::chrono::steady_clock::now();
        } 
        // wait k querys
        else if (querys_it->second.size() < k_) {
            querys_it->second.emplace_back(infer_query);
            // if(!infer_query->is_parity_data_) {
            //     ElasticcdcReply reply;
            //     std::string reply_data(infer_query->reply_info_bytes.begin()+ DATASETS.at(model_name_).first,
            //                                     infer_query->reply_info_bytes.end());
            //     reply.set_reply_info(reply_data);
            //     reply.set_id(infer_query->id_ - start_task_id);
            //     reply.set_recompute(infer_query->is_recompute_);
            //     if (infer_query->stream_) {
            //         LOG_INFO("Send query %d to client as %d.", infer_query->id_, infer_query->id_ - start_task_id);
            //         infer_query->stream_->Write(reply);
            //         tasks_completed_num++;
            //     }
            // }
        } 

        querys_it = querys_id1.find(decode_id);
        if(querys_it->second.size() == k_ && monitor_->IsStripeBroken(decode_id)){
            assert(stripes.find(decode_id) != stripes.end());
            auto stripe_ids = stripes[decode_id];

            // std::cout << "decode id: " << decode_id << " stripe_ids.size: " << stripe_ids.size() << std::endl;
            bool is_need_decode = false;
            std::vector<std::vector<uint8_t>> data;
            // data.emplace_back(infer_query->reply_info_bytes);
            std::shared_ptr<grpcStream> stream;
            // int64_t id;
            double infer_time;
            bool is_recompute;

            for (const auto& next_query : querys_it->second) {
                SingleQuery* singleQuery = dynamic_cast<SingleQuery*>(next_query);
                stripe_ids.erase(singleQuery->id_);
                if (conf_->model_name.find("irevnet") != std::string::npos)
                    data.emplace_back(std::vector<uint8_t>(singleQuery->reply_info_bytes.begin(),
                                                         singleQuery->reply_info_bytes.begin() + DATASETS.at(model_name_).first));
                else 
                    data.emplace_back(std::vector<uint8_t>(singleQuery->reply_info_bytes.begin(),
                                                         singleQuery->reply_info_bytes.end()));
                if (!singleQuery->is_parity_data_) {
                    // ElasticcdcReply reply;
                    // std::string reply_data(singleQuery->reply_info_bytes.begin()+ DATASETS.at(model_name_).first,
                    //                         singleQuery->reply_info_bytes.end());
                    // reply.set_reply_info(reply_data);
                    // reply.set_id(singleQuery->id_ - start_task_id);
                    // reply.set_recompute(singleQuery->is_recompute_);
                    // if (singleQuery->stream_) {
                    //     singleQuery->stream_->Write(reply);
                    //     tasks_completed_num++;
                    // }
                } else {
                    is_need_decode = true;
                    stream = singleQuery->stream_;
                    // id = singleQuery->id_;
                    infer_time = singleQuery -> infer_time;
                    is_recompute = singleQuery->is_recompute_;

                }
            }
            if (is_need_decode && stream) {
                assert(stripe_ids.size() == 1);
                auto id = *stripe_ids.cbegin();
                std::cout << "broken id: " << id << std::endl;
                std::vector<uint8_t> reply_info;
                auto start11 = std::chrono::high_resolution_clock::now();
                // std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout << "decoder performed ~" << std::endl;
                decoder_->decode(data, reply_info);
                // std::cout << "decode completed" << std::endl;
                auto end11 = std::chrono::high_resolution_clock::now();

                auto duration11 = std::chrono::duration_cast<std::chrono::milliseconds>(end11 - start11).count();

                LOG_INFO("decode time: %ld ms", duration11);

                // update infer_time and decode_time
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                                (end - decode_start_time[decode_id]).count();

                auto end1 = std::chrono::steady_clock::now();
                auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - decode_start_time1[decode_id]);
                {
                    std::unique_lock<std::mutex> lock(*mtx_);
                    cdc_infer_time_ = infer_time;
                    decode_time_ = duration;

                    LOG_INFO("Update cdc infer time:%lf, decode id: %d, decode time:%lf, decode time1:%ld", cdc_infer_time_, decode_id, decode_time_, duration1.count());
                }

                filter_->updateFilterRatio(duration, 0);

                ElasticcdcReply reply;
                std::string reply_data(reply_info.begin(), reply_info.end());
                reply.set_reply_info(reply_data);
                reply.set_id(id - start_task_id);
                reply.set_recompute(is_recompute);
                
                std::unique_lock<std::mutex> task_lock(*taskCountMtx);
                if(!is_stripes_completed[decode_id]) {
                    stream->Write(reply);
                    tasks_completed_num++;
                    is_stripes_completed[decode_id] = true;
                    LOG_INFO("Send decoded query %d to client as %d.", id, id - start_task_id);
                }
                

            }
            // visited_id1.insert(decode_id);
            querys_id1.erase(decode_id);
            decode_start_time.erase(decode_id);
            {
                std::unique_lock<std::mutex> lock(*mtx_stripes_);
                stripes.unsafe_erase(decode_id);
            }
        }
    // }
}

/**
 * handle CDC reply
 */
void DecodeWorker::handleCDCQueryFlagAlgBaseline(std::unordered_map<int, std::vector<Query*>>& querys_id1,
                                        std::unordered_set<int>& visited_id1,
                                        SingleQuery* infer_query,
                                        std::unordered_map<int, 
                                        std::chrono::high_resolution_clock::time_point>& decode_start_time,
                                        std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>>& decode_start_time1) {
    LOG_INFO("handle CDC query with Flag Alg Baseline!");
    int decode_id = infer_query->encode_id_;
    // auto visited1_it = visited_id1.find(decode_id);
    
    // the decoding task of decode_id is not over
    // if (visited1_it == visited_id1.end()) {
        auto querys_it = querys_id1.find(decode_id);
        // first receive the query of the decode_id
        if (querys_it == querys_id1.end()) {
            querys_id1[decode_id] = {infer_query};
            // if(!infer_query->is_parity_data_) {
            //     ElasticcdcReply reply;
            //     std::string reply_data(infer_query->reply_info_bytes.begin()+ DATASETS.at(model_name_).first,
            //                                     infer_query->reply_info_bytes.end());
            //     reply.set_reply_info(reply_data);
            //     reply.set_id(infer_query->id_ - start_task_id);
            //     reply.set_recompute(infer_query->is_recompute_);
            //     if (infer_query->stream_) {
            //         LOG_INFO("Send query %d to client as %d.", infer_query->id_, infer_query->id_ - start_task_id);
            //         infer_query->stream_->Write(reply);
            //         tasks_completed_num++;
            //     }
            // }
            decode_start_time[decode_id] = std::chrono::high_resolution_clock::now();
            decode_start_time1[decode_id] = std::chrono::steady_clock::now();
        } 
        // wait k querys
        else if (querys_it->second.size() < k_) {
            querys_it->second.emplace_back(infer_query);
            // if(!infer_query->is_parity_data_) {
            //     ElasticcdcReply reply;
            //     std::string reply_data(infer_query->reply_info_bytes.begin()+ DATASETS.at(model_name_).first,
            //                                     infer_query->reply_info_bytes.end());
            //     reply.set_reply_info(reply_data);
            //     reply.set_id(infer_query->id_ - start_task_id);
            //     reply.set_recompute(infer_query->is_recompute_);
            //     if (infer_query->stream_) {
            //         LOG_INFO("Send query %d to client as %d.", infer_query->id_, infer_query->id_ - start_task_id);
            //         infer_query->stream_->Write(reply);
            //         tasks_completed_num++;
            //     }
            // }
        } 

        querys_it = querys_id1.find(decode_id);
        if(querys_it->second.size() == k_){
            assert(stripes.find(decode_id) != stripes.end());
            auto stripe_ids = stripes[decode_id];

            // std::cout << "decode id: " << decode_id << " stripe_ids.size: " << stripe_ids.size() << std::endl;
            bool is_need_decode = false;
            std::vector<std::vector<uint8_t>> data;
            // data.emplace_back(infer_query->reply_info_bytes);
            std::shared_ptr<grpcStream> stream;
            // int64_t id;
            double infer_time;
            bool is_recompute;

            for (const auto& next_query : querys_it->second) {
                SingleQuery* singleQuery = dynamic_cast<SingleQuery*>(next_query);
                stripe_ids.erase(singleQuery->id_);
                if (conf_->model_name.find("irevnet") != std::string::npos)
                    data.emplace_back(std::vector<uint8_t>(singleQuery->reply_info_bytes.begin(),
                                                         singleQuery->reply_info_bytes.begin() + DATASETS.at(model_name_).first));
                else
                    data.emplace_back(std::vector<uint8_t>(singleQuery->reply_info_bytes.begin(),
                                                         singleQuery->reply_info_bytes.end()));
                if (!singleQuery->is_parity_data_) {
                    // ElasticcdcReply reply;
                    // std::string reply_data(singleQuery->reply_info_bytes.begin()+ DATASETS.at(model_name_).first,
                    //                         singleQuery->reply_info_bytes.end());
                    // reply.set_reply_info(reply_data);
                    // reply.set_id(singleQuery->id_ - start_task_id);
                    // reply.set_recompute(singleQuery->is_recompute_);
                    // if (singleQuery->stream_) {
                    //     singleQuery->stream_->Write(reply);
                    //     tasks_completed_num++;
                    // }
                } else {
                    is_need_decode = true;
                    stream = singleQuery->stream_;
                    // id = singleQuery->id_;
                    infer_time = singleQuery -> infer_time;
                    is_recompute = singleQuery->is_recompute_;

                }
            }
            if (is_need_decode && stream) {
                assert(stripe_ids.size() == 1);
                auto id = *stripe_ids.cbegin();
                std::cout << "broken id: " << id << std::endl;
                std::vector<uint8_t> reply_info;
                auto start11 = std::chrono::high_resolution_clock::now();
                // std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout << "decoder performed ~" << std::endl;
                decoder_->decode(data, reply_info);
                std::cout << "decode completed" << std::endl;
                auto end11 = std::chrono::high_resolution_clock::now();

                auto duration11 = std::chrono::duration_cast<std::chrono::milliseconds>(end11 - start11).count();

                LOG_INFO("decode time: %ld ms", duration11);

                // update infer_time and decode_time
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                                (end - decode_start_time[decode_id]).count();

                auto end1 = std::chrono::steady_clock::now();
                auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - decode_start_time1[decode_id]);
                {
                    std::unique_lock<std::mutex> lock(*mtx_);
                    cdc_infer_time_ = infer_time;
                    decode_time_ = duration;

                    LOG_INFO("Update cdc infer time:%lf, decode id: %d, decode time:%lf, decode time1:%ld", cdc_infer_time_, decode_id, decode_time_, duration1.count());
                }

                filter_->updateFilterRatio(duration, 0);

                ElasticcdcReply reply;
                std::string reply_data(reply_info.begin(), reply_info.end());
                reply.set_reply_info(reply_data);
                reply.set_id(id - start_task_id);
                reply.set_recompute(is_recompute);
                
                std::unique_lock<std::mutex> task_lock(*taskCountMtx);
                if(!is_stripes_completed[decode_id]) {
                    stream->Write(reply);
                    tasks_completed_num++;
                    is_stripes_completed[decode_id] = true;
                    LOG_INFO("Send decoded query %d to client as %d.", id, id - start_task_id);
                }
                

            }
            // visited_id1.insert(decode_id);
            querys_id1.erase(decode_id);
            decode_start_time.erase(decode_id);
            {
                std::unique_lock<std::mutex> lock(*mtx_stripes_);
                stripes.unsafe_erase(decode_id);
            }
        }
    // }
}

void DecodeWorker::handleCDCQueryBaseline(std::unordered_map<int, std::vector<Query*>>& querys_id1,
                                            std::unordered_set<int>& visited_id1,
                                            SingleQuery* infer_query,
                                            std::unordered_map<int, 
                                            std::chrono::high_resolution_clock::time_point>& decode_start_time,
                                            std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>>& decode_start_time1) {

    LOG_INFO("Baseline handle CDC query!");
    int decode_id = infer_query->encode_id_;
    // auto visited1_it = visited_id1.find(decode_id);

    // the decoding task of decode_id is not over
    // if (visited1_it == visited_id1.end()) {
        auto querys_it = querys_id1.find(decode_id);
        // first receive the query of the decode_id
        if (querys_it == querys_id1.end()) {
            querys_id1[decode_id] = {infer_query};
            // if(!infer_query->is_parity_data_) {
            //     ElasticcdcReply reply;
            //     std::string reply_data(infer_query->reply_info_bytes.begin()+ DATASETS.at(model_name_).first,
            //                                     infer_query->reply_info_bytes.end());
            //     reply.set_reply_info(reply_data);
            //     reply.set_id(infer_query->id_ - start_task_id);
            //     reply.set_recompute(infer_query->is_recompute_);
            //     if (infer_query->stream_) {
            //         infer_query->stream_->Write(reply);
            //         tasks_completed_num++;
            //     }
            // }
            // data.emplace_back(std::vector<uint8_t>(infer_query->reply_info_bytes.begin(),
            //                                         infer_query->reply_info_bytes.begin() + DATASETS.at(model_name_).first));
            
            decode_start_time[decode_id] = std::chrono::high_resolution_clock::now();
            decode_start_time1[decode_id] = std::chrono::steady_clock::now();
        } 
        // wait k querys
        else if (querys_it->second.size() < k_) {
            querys_it->second.emplace_back(infer_query);

            // if(!infer_query->is_parity_data_) {
            //     ElasticcdcReply reply;
            //     std::string reply_data(infer_query->reply_info_bytes.begin()+ DATASETS.at(model_name_).first,
            //                                     infer_query->reply_info_bytes.end());
            //     reply.set_reply_info(reply_data);
            //     reply.set_id(infer_query->id_ - start_task_id);
            //     reply.set_recompute(infer_query->is_recompute_);
            //     if (infer_query->stream_) {
            //         infer_query->stream_->Write(reply);
            //         tasks_completed_num++;
            //     }
            // }
            // data.emplace_back(std::vector<uint8_t>(infer_query->reply_info_bytes.begin(),
            //                                         infer_query->reply_info_bytes.begin() + DATASETS.at(model_name_).first));
        } 

        querys_it = querys_id1.find(decode_id);
        if(querys_it->second.size() == k_){
            auto stripe_ids = stripes[decode_id];
            bool is_need_reply = false;
            std::vector<std::vector<uint8_t>> data;
            // data.emplace_back(infer_query->reply_info_bytes);
            std::shared_ptr<grpcStream> stream;
            // int64_t id;
            double infer_time;
            bool is_recompute;

            for (const auto& next_query : querys_it->second) {
                SingleQuery* singleQuery = dynamic_cast<SingleQuery*>(next_query);
                stripe_ids.erase(singleQuery->id_);
                if (conf_->model_name.find("irevnet") != std::string::npos)
                    data.emplace_back(std::vector<uint8_t>(singleQuery->reply_info_bytes.begin(),
                                                        singleQuery->reply_info_bytes.begin() + DATASETS.at(model_name_).first));
                else
                    data.emplace_back(std::vector<uint8_t>(singleQuery->reply_info_bytes.begin(),
                                                        singleQuery->reply_info_bytes.end()));
                if (!singleQuery->is_parity_data_) {
                    // ElasticcdcReply reply;
                    // std::string reply_data(singleQuery->reply_info_bytes.begin()+ DATASETS.at(model_name_).first,
                    //                         singleQuery->reply_info_bytes.end());
                    // reply.set_reply_info(reply_data);
                    // reply.set_id(singleQuery->id_ - start_task_id);
                    // reply.set_recompute(singleQuery->is_recompute_);
                    // if (singleQuery->stream_) {
                    //     singleQuery->stream_->Write(reply);
                    //     tasks_completed_num++;
                    // }
                } else {
                    is_need_reply = true;
                    stream = singleQuery->stream_;
                    is_recompute = singleQuery->is_recompute_;
                    // id = singleQuery->id_;
                }
                infer_time = singleQuery -> infer_time;
            }

            std::vector<uint8_t> reply_info;
            auto start11 = std::chrono::high_resolution_clock::now();
            std::cout << "decoder performed ~" << std::endl;
            decoder_->decode(data, reply_info);
            auto end11 = std::chrono::high_resolution_clock::now();

            auto duration11 = std::chrono::duration_cast<std::chrono::milliseconds>(end11 - start11).count();

            LOG_INFO("decode time: %ld ms", duration11);

            // update infer_time and decode_time
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - decode_start_time[decode_id]).count();

            auto end1 = std::chrono::steady_clock::now();
            auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - decode_start_time1[decode_id]);
            {
                std::unique_lock<std::mutex> lock(*mtx_);
                cdc_infer_time_ = infer_time;
                decode_time_ = duration;

                LOG_INFO("Update cdc infer time:%lf, decode id: %d, decode time:%lf, decode time1:%ld", cdc_infer_time_, decode_id, decode_time_, duration1.count());
            }

            filter_->updateFilterRatio(duration, 0);

            if(is_need_reply && stream) {
                assert(stripe_ids.size() == 1);
                auto id = *stripe_ids.cbegin();
                ElasticcdcReply reply;
                std::string reply_data(reply_info.begin(), reply_info.end());
                reply.set_reply_info(reply_data);
                reply.set_id(id - start_task_id);
                reply.set_recompute(is_recompute);
                std::unique_lock<std::mutex> task_lock(*taskCountMtx);
                if(!is_stripes_completed[decode_id]) {
                    stream->Write(reply);
                    tasks_completed_num++;
                    is_stripes_completed[decode_id] = true;
                }
            }
            
            // visited_id1.insert(decode_id);
            querys_id1.erase(decode_id);
            decode_start_time.erase(decode_id);
            {
                std::unique_lock<std::mutex> lock(*mtx_stripes_);
                stripes.unsafe_erase(decode_id);
            }
        // }
    }
}

/**
 * handle backup reply
 */
void DecodeWorker::handleBackupQuery(std::unordered_set<int>& visited_id2,
                                     SingleQuery* infer_query) {
    LOG_INFO("handle Backup query!");
    auto visited2_it = visited_id2.find(infer_query->encode_id_);

    if (visited2_it == visited_id2.end()) {
        auto backup_id = backups[infer_query->encode_id_];
        ElasticcdcReply reply;

        std::vector<std::string> reply_data_vec;
            if (conf_->model_name.find("irevnet") != std::string::npos) {
                std::string reply_data(infer_query->reply_info_bytes.begin() + DATASETS.at(model_name_).first,
                                    infer_query->reply_info_bytes.end());
                reply_data_vec.push_back(reply_data);
            }
            else {
                std::string reply_data(infer_query->reply_info_bytes.begin(),
                                    infer_query->reply_info_bytes.end());
                reply_data_vec.push_back(reply_data);
            }
        // std::string reply_data = "";
        reply.set_reply_info(reply_data_vec[0]);
        // if(!infer_query -> is_parity_data_) 
        //     reply.set_id(infer_query->id_ - start_task_id);
        // else reply.set_id(infer_query->id_);
        reply.set_id(backup_id - start_task_id);
        reply.set_recompute(infer_query->is_recompute_);
        if (infer_query->stream_) {
            LOG_INFO("Send query %d to client as %d.", backup_id, backup_id - start_task_id);
            std::unique_lock<std::mutex> task_lock(*taskCountMtx);
            infer_query->stream_->Write(reply);
            tasks_completed_num++;
        }
        visited_id2.insert(infer_query->encode_id_);
        {
            std::unique_lock<std::mutex> lock(*mtx_backups_);
            backups.unsafe_erase(infer_query->encode_id_);
        }
    }
}

void DecodeWorker::handleBrokenQuery(std::unordered_map<int, std::vector<Query*>>& querys_id1,
                                    std::unordered_set<int>& visited_id1,
                                    SingleQuery* infer_query,
                                    std::unordered_map<int, 
                                        std::chrono::high_resolution_clock::time_point>& decode_start_time) {
    for(auto& info: encode_fail_num) {
       
        auto decode_id = info.first;
        auto fail_num = info.second.first;
        auto total_receive_num = info.second.second;

        if(total_receive_num == conf_->k + 1 && fail_num > 1 && querys_id1.find(decode_id) != querys_id1.end()) {
            LOG_INFO("handle broken query, decode id: %ld, fail num: %d", decode_id, fail_num);


            querys_id1.erase(decode_id); 
            decode_start_time.erase(decode_id);
            {
                std::lock_guard<std::mutex> lock(*mtx_encode_fail_num_);
                encode_fail_num.unsafe_erase(decode_id);
            }
            {
                std::unique_lock<std::mutex> lock(*mtx_stripes_);
                stripes.unsafe_erase(decode_id);
            }
        }
    }
}

void DecodeWorker::handleLastQueryBroken(std::unordered_map<int, std::vector<Query*>>& querys_id1,
                                    std::unordered_map<int, 
                                        std::chrono::high_resolution_clock::time_point>& decode_start_time) {
for(auto& info: encode_fail_num) {
       
        auto decode_id = info.first;
        auto total_receive_num = info.second.second;

        if(!is_stripes_completed[decode_id] && monitor_->IsStripeBroken(decode_id)
            && querys_id1.find(decode_id) != querys_id1.end() && querys_id1[decode_id].size() == conf_->k) {
            assert(stripes.find(decode_id) != stripes.end());
            auto stripe_ids = stripes[decode_id];

            // std::cout << "decode id: " << decode_id << " stripe_ids.size: " << stripe_ids.size() << std::endl;
            bool is_need_decode = false;
            std::vector<std::vector<uint8_t>> data;
            std::shared_ptr<grpcStream> stream;
            double infer_time;
            bool is_recompute;

            auto querys_it = querys_id1.find(decode_id);
            for (const auto& next_query : querys_it->second) {
                SingleQuery* singleQuery = dynamic_cast<SingleQuery*>(next_query);
                stripe_ids.erase(singleQuery->id_);
                if (conf_->model_name.find("irevnet") != std::string::npos)
                    data.emplace_back(std::vector<uint8_t>(singleQuery->reply_info_bytes.begin(),
                                                         singleQuery->reply_info_bytes.begin() + DATASETS.at(model_name_).first));
                else
                    data.emplace_back(std::vector<uint8_t>(singleQuery->reply_info_bytes.begin(),
                                                         singleQuery->reply_info_bytes.end()));
                if (singleQuery->is_parity_data_) {
                    is_need_decode = true;
                    stream = singleQuery->stream_;
                    // id = singleQuery->id_;
                    infer_time = singleQuery -> infer_time;
                    is_recompute = singleQuery->is_recompute_;

                }
            }
            if (is_need_decode && stream) {
                assert(stripe_ids.size() == 1);
                auto id = *stripe_ids.cbegin();
                std::cout << "broken id: " << id << std::endl;
                std::vector<uint8_t> reply_info;
                auto start11 = std::chrono::high_resolution_clock::now();
                std::cout << "decoder performed ~" << std::endl;
                decoder_->decode(data, reply_info);
                std::cout << "decode completed" << std::endl;
                auto end11 = std::chrono::high_resolution_clock::now();

                auto duration11 = std::chrono::duration_cast<std::chrono::milliseconds>(end11 - start11).count();

                LOG_INFO("decode time: %ld ms", duration11);

                // update infer_time and decode_time
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>
                                (end - decode_start_time[decode_id]).count();
                {
                    std::unique_lock<std::mutex> lock(*mtx_);
                    cdc_infer_time_ = infer_time;
                    decode_time_ = duration;

                    LOG_INFO("Update cdc infer time:%lf, decode id: %ld, decode time:%lf", cdc_infer_time_, decode_id, decode_time_);
                }

                filter_->updateFilterRatio(duration, 0);

                ElasticcdcReply reply;
                std::string reply_data(reply_info.begin(), reply_info.end());
                reply.set_reply_info(reply_data);
                reply.set_id(id - start_task_id);
                reply.set_recompute(is_recompute);
                
                std::unique_lock<std::mutex> task_lock(*taskCountMtx);
                if(!is_stripes_completed[decode_id]) {
                    stream->Write(reply);
                    tasks_completed_num++;
                    is_stripes_completed[decode_id] = true;
                    LOG_INFO("Send decoded query %d to client as %d.", id, id - start_task_id);
                }
                

            }
            // visited_id1.insert(decode_id);
            querys_id1.erase(decode_id);
            decode_start_time.erase(decode_id);
            {
                std::lock_guard<std::mutex> lock(*mtx_encode_fail_num_);
                encode_fail_num.unsafe_erase(decode_id);
            }
            {
                std::unique_lock<std::mutex> lock(*mtx_stripes_);
                stripes.unsafe_erase(decode_id);
            }
        }
    }
}