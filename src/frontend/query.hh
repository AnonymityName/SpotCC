#pragma once
#include "../inc/inc.hh"
#include <iostream>
#include <memory>
#include <string>
#include "../common/image.hh"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "../protocol/elasticcdc.pb.h"
#include "../protocol/elasticcdc.grpc.pb.h"


class Query {
public:
    std::string model_name_;
    std::string scale_;
    std::string filename_;
    int id_;
    std::condition_variable cv_;
    std::mutex mutex_;
    // std::string reply_info_;
    int encode_id_;
    std::string encode_type_;
    bool is_parity_data_ = false;
    bool is_recompute_ = false; 

    virtual ~Query() {}
};
    
class SingleQuery: public Query {
public:
    SingleQuery(const ImageClassifyArgs& request_info) {
        model_name_ = request_info.model_name;
        scale_ = request_info.scale;
        filename_ = request_info.filename;
        id_ = request_info.id;
        data_.assign(request_info.data.begin(),request_info.data.end());
        stream_ = request_info.stream;
        encode_id_ = request_info.encode_id;
        encode_type_ = request_info.encode_type;
        end_signal_ = request_info.end_signal_;
        is_recompute_ = request_info.is_recompute_;
        is_parity_data_ = request_info.is_parity_data_;
    }
    std::shared_ptr<grpcStream> stream_;
    std::vector<uint8_t> data_;
    std::string reply_info_;
    std::vector<uint8_t> reply_info_bytes;
    bool end_signal_;
    double infer_time;
};

class BatchQuery: public Query {
public:
    BatchQuery(std::string model_name, 
                std::string scale, 
                std::string filename, 
                int id, 
                std::vector<std::vector<uint8_t>> data,
                std::vector<std::shared_ptr<grpcStream>> streams) 
    {
        model_name_ = model_name;
        scale_ = scale;
        filename_ = filename;
        id_ = id;
        data_ = data;
        batch_size_ = data.size();
        streams_ = streams;
    }
    std::vector<std::vector<uint8_t>> data_;
    std::vector<std::shared_ptr<grpcStream>> streams_;
    int batch_size_;
    std::vector<std::string> reply_info_;
};
