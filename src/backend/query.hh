#pragma once
#include "../inc/inc.hh"
#include <iostream>
#include <memory>
#include <string>
#include "image_classify.hh"

class Query {
public:
    std::string model_name_;
    std::string scale_;
    std::string filename_;
    int id_;
    std::condition_variable cv_;
    std::mutex mutex_;
    std::string encode_type_;
    int front_id_;
    bool end_signal_;
    bool recompute_;
    // std::string reply_info_;
};
    
class SingleQuery: public Query {
public:
    SingleQuery(std::string model_name,
                std::string scale,
                std::string filename,
                int id,
                std::vector<uint8_t> data,
                std::string encode_type,
                grpcStream* stream,
                int front_id,
                bool end_signal,
                bool recompute) {
        model_name_ = model_name;
        scale_ = scale;
        filename_ = filename;
        id_ = id;
        data_ = data;
        encode_type_ = encode_type;
        stream_ = stream;
        front_id_ = front_id;
        end_signal_ = end_signal;
        recompute_ = recompute;
    }
    // SingleQuery(const ImageArgs& request) {
    //     model_name_ = request.model_name;
    //     scale_ = request.scale;
    //     filename_ = request.filename;
    //     id_ = request.id;
    //     data_ = request.data;
    //     encode_type_ = request.encode_type;
    //     stream_ = request.stream;
    //     front_id_ = request.front_id;
    //     end_signal_ = request.end_signal;
    // }
    grpcStream* stream_;
    std::vector<uint8_t> data_;
    std::string reply_info_;
};

class BatchQuery: public Query {
public:
    BatchQuery(std::string model_name, 
                std::string scale,
                std::vector<std::string> filenames, 
                int id, 
                std::vector<std::vector<uint8_t>> data,
                std::vector<grpcStream*> streams,
                std::string encode_type,
                std::vector<int> ids) 
    {
        model_name_ = model_name;
        scale_ = scale;
        filenames_ = filenames;
        id_ = id;
        data_ = data;
        batch_size_ = data.size();
        streams_ = streams;
        encode_type_ = encode_type;
        ids_ = ids;
    }
    std::vector<int> ids_;
    std::vector<std::string> filenames_;
    std::vector<std::vector<uint8_t>> data_;
    std::vector<grpcStream*> streams_;
    int batch_size_;
    std::vector<std::string> reply_info_;
};