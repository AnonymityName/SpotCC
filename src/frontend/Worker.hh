#pragma once
#include "../inc/inc.hh"
#include "../common/concurrency_queue.hh"
#include "../common/conf.hh"
#include "../common/image.hh"
#include "../common/logger.hh"
#include "../common/concurrency_set.hh"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "filter.hh"
#include "encoder.hh"
#include "decoder.hh"
#include "query.hh"
#include "preprocessor.hh"
// #include "monitor.hh"
#include "monitor2parts.hh"
#include "../protocol/elasticcdc.pb.h"
#include "../protocol/elasticcdc.grpc.pb.h"
#include <google/protobuf/empty.pb.h>
#include <chrono>
// #include <../common/thread_pool.hh>

//need to download
#include <tbb/concurrent_unordered_map.h>


#define ENCODE_DATA_ID_MIN 1500000000

using grpc::ClientContext;
using grpc::ClientReaderWriter;
using grpc::ServerReaderWriter;
using grpc::Channel;

using QueryQueue = ConcurrencyQueue<Query*>;
using QueryIdSet = ConcurrencySet<uint32_t>;

extern int tasks_completed_num;
extern int start_task_id;
extern std::condition_variable completedCV;
extern std::mutex completedMutex;
extern std::shared_ptr<std::mutex> taskCountMtx;

extern int task_num;

class Worker {
public:
    virtual void run() = 0;
    virtual void init() = 0;

    Worker(std::shared_ptr<Config> conf_, 
        std::shared_ptr<QueryQueue> queue_1,
        std::shared_ptr<std::mutex> mtx_1,
        std::shared_ptr<std::condition_variable> cv_1,
        std::shared_ptr<QueryQueue> queue_2,
        std::shared_ptr<std::mutex> mtx_2,
        std::shared_ptr<std::condition_variable> cv_2,
        std::shared_ptr<Monitor> monitor):
        conf_(conf_), 
        queue_1_(queue_1), 
        mtx_1_(mtx_1), 
        cv_1_(cv_1), 
        queue_2_(queue_2), 
        mtx_2_(mtx_2), 
        cv_2_(cv_2),
        monitor_(monitor) {};

    std::shared_ptr<Config> conf_;
    std::shared_ptr<Monitor> monitor_;

    std::shared_ptr<QueryQueue> queue_1_;
    std::shared_ptr<std::mutex> mtx_1_;
    std::shared_ptr<std::condition_variable> cv_1_;

    std::shared_ptr<QueryQueue> queue_2_;
    std::shared_ptr<std::mutex> mtx_2_;
    std::shared_ptr<std::condition_variable> cv_2_;

    static std::shared_ptr<std::mutex> mtx_;
    static double cdc_infer_time_;
    static double backup_infer_time_;
    static double decode_time_;

    static std::shared_ptr<std::mutex> mtx_encode_fail_num_;
    static std::shared_ptr<std::mutex> mtx_stripes_;
    static std::shared_ptr<std::mutex> mtx_backups_;
    static tbb::concurrent_unordered_map<uint64_t, std::pair<uint32_t, uint32_t>>  encode_fail_num; //[encode_id, <fail_num, total_num>]
    static tbb::concurrent_unordered_map<uint64_t, std::unordered_set<uint32_t>>  stripes; //[encode_id, [id]]
    static tbb::concurrent_unordered_map<uint64_t, uint32_t>  backups; //[encode_id, id]
    static tbb::concurrent_unordered_map<uint64_t, bool>  is_stripes_completed; //[encode_id, is_completed]
};

class PreprocessWorker: private Worker {
public:
    PreprocessWorker(std::shared_ptr<Config> conf_,
                std::shared_ptr<QueryQueue> queue_1,
                std::shared_ptr<std::mutex> mtx_1,
                std::shared_ptr<std::condition_variable> cv_1,
                std::shared_ptr<QueryQueue> queue_2,
                std::shared_ptr<std::mutex> mtx_2,
                std::shared_ptr<std::condition_variable> cv_2,
                std::shared_ptr<Monitor> monitor);
    ~PreprocessWorker();

    void run() override;
    void init() override;

private:
    std::thread pp_thread_;

    std::string format;
    std::string dtype;
    u_int32_t channels;
    u_int32_t height;
    u_int32_t width;
};

class EncodeWorker: private Worker {
public:
    EncodeWorker(std::shared_ptr<Config> conf_,
                std::shared_ptr<QueryQueue> queue_1,
                std::shared_ptr<std::mutex> mtx_1,
                std::shared_ptr<std::condition_variable> cv_1,
                std::shared_ptr<QueryQueue> queue_2,
                std::shared_ptr<std::mutex> mtx_2,
                std::shared_ptr<std::condition_variable> cv_2,
                std::shared_ptr<Monitor> monitor,
                std::shared_ptr<Filter> filter);
    ~EncodeWorker();

    void run() override;
    void init() override;

private:
    std::thread encode_thread_;
    uint32_t backup_num_;
    uint32_t k_ ;

    std::shared_ptr<Encoder> encoder_;
    std::shared_ptr<Filter> filter_;

    ImageClassifyArgs createRequestInfo(const SingleQuery& query,
                                         const std::string& encodeType,
                                         int encode_id, bool is_parity_data,
                                         bool is_end_signal,
                                         bool is_recompute);

}; 

class InferWorker: private Worker {
public:
    InferWorker(std::shared_ptr<Config> conf,
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
                std::uint32_t frontend_id);
    ~InferWorker();

    void run() override;
    void init() override;

private:
    std::thread infer_thread_;
    std::vector<std::thread> recv_threads_;
    uint32_t node_number_ = 1;
    tbb::concurrent_unordered_map<uint64_t, Query*> querys_map_;
    tbb::concurrent_unordered_map<uint64_t,
     std::chrono::high_resolution_clock::time_point> querys_start_time_map_;
    std::uint32_t frontend_id_;

    std::shared_ptr<QueryQueue> queue_3_;
    std::shared_ptr<std::mutex> mtx_3_;
    std::shared_ptr<std::condition_variable> cv_3_;
    tbb::concurrent_unordered_map<uint64_t, std::vector<Query*>> querys_encode_map_;
    tbb::concurrent_unordered_map<uint64_t, std::vector<Query*>> querys_backup_map_;
    tbb::concurrent_unordered_map<uint64_t, uint32_t>  backup_fail_num; //[encode_id, fail_num]

    std::shared_ptr<std::mutex> mtx_map_;
    std::shared_ptr<std::mutex> mtx_start_time_map_;
    std::shared_ptr<std::mutex> mtx_querys_encode_map_;
    std::shared_ptr<std::mutex> mtx_querys_backup_map_;
    std::shared_ptr<std::mutex> mtx_backup_fail_num_;

    uint32_t query_num = 0;

    std::shared_ptr<std::mutex> mtx_visited_id1_;
    std::shared_ptr<std::mutex> mtx_visited_id2_;
    std::shared_ptr<QueryIdSet> visited_id1_;    
    std::shared_ptr<QueryIdSet> visited_id2_;
    tbb::concurrent_unordered_map<uint64_t, uint32_t> received_querys_num_; //[encode_id, num]

    std::string model_name_;

    void ReceiveResponses(std::shared_ptr<grpcStreamClient> stream, std::string backend_ip);
    void SendToClient(SingleQuery* recv_query);

}; 

class DecodeWorker: private Worker {
public:
    DecodeWorker(std::shared_ptr<Config> conf_,
                std::shared_ptr<QueryQueue> queue_1,
                std::shared_ptr<std::mutex> mtx_1,
                std::shared_ptr<std::condition_variable> cv_1,
                std::shared_ptr<QueryQueue> queue_2,
                std::shared_ptr<std::mutex> mtx_2,
                std::shared_ptr<std::condition_variable> cv_2,
                std::shared_ptr<Monitor> monitor,
                std::shared_ptr<Filter> filter);
    ~DecodeWorker();

    void run() override;
    void init() override;

private:  
    std::thread decode_thread_;
    std::thread background_thread_;
    uint32_t k_ ;
    std::shared_ptr<Decoder> decoder_;
    std::shared_ptr<Filter> filter_;
    std::string model_name_;


    void handleCDCQuery(std::unordered_map<int, std::vector<Query*>>& querys_id1,
                        std::unordered_set<int>& visited_id1,
                        SingleQuery* infer_query,
                        std::unordered_map<int,std::chrono::high_resolution_clock::time_point>& decode_start_time,
                        std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>>& decode_start_time1);

    void handleCDCQueryBaseline(std::unordered_map<int, std::vector<Query*>>& querys_id1,
                                std::unordered_set<int>& visited_id1,
                                SingleQuery* infer_query,
                                std::unordered_map<int, 
                                std::chrono::high_resolution_clock::time_point>& decode_start_time,
                                std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>>& decode_start_time1);

    void handleCDCQueryFlagAlgBaseline(std::unordered_map<int, std::vector<Query*>>& querys_id1,
                                        std::unordered_set<int>& visited_id1,
                                        SingleQuery* infer_query,
                                        std::unordered_map<int, 
                                        std::chrono::high_resolution_clock::time_point>& decode_start_time,
                                        std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>>& decode_start_time1);

    void handleBackupQuery(std::unordered_set<int>& visited_id2,
                            SingleQuery* infer_query);

    void handleBrokenQuery(std::unordered_map<int, std::vector<Query*>>& querys_id1,
                        std::unordered_set<int>& visited_id1,
                        SingleQuery* infer_query,
                        std::unordered_map<int, std::chrono::high_resolution_clock::time_point>& decode_start_time);

    void handleLastQueryBroken(std::unordered_map<int, std::vector<Query*>>& querys_id1,
                                std::unordered_map<int, 
                                    std::chrono::high_resolution_clock::time_point>& decode_start_time);
                                    


};