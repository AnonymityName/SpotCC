#include <iostream>
#include <memory>
#include <string>
#include "../inc/inc.hh"

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#ifdef BAZEL_BUILD
#include "examples/protos/elasticcdc.grpc.pb.h"
#else
#include "../protocol/elasticcdc.grpc.pb.h"
#endif

#include "image_classify.hh"
#include "../common/logger.hh"
#include "backend.hh"

using RequestQueue = std::queue<std::pair<ElasticcdcRequest, grpcStream*>>;

RequestQueue requestQueue;
std::mutex queueMutex;
std::condition_variable queueCV;

std::mutex m;

void processRequests(const std::string conf_path) {
    auto backend_ = std::make_shared<Backend>(conf_path);
    while(true) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCV.wait(lock, [] { return !requestQueue.empty(); });

        auto [request, stream] = requestQueue.front();
        LOG_INFO("pop request %ld from the requestQueue", request.id());
        requestQueue.pop();
        lock.unlock();

        // send reply
        ImageArgs request_info;
        request_info.filename = request.filename();
        request_info.model_name = request.model_name();
        request_info.id = request.id();
        request_info.scale = request.scale();
        request_info.data = request.data();
        request_info.stream = stream;
        request_info.cdc_infer_time = request.cdc_infer_time();
        request_info.backup_infer_time = request.backup_infer_time();
        request_info.decode_time = request.decode_time();
        request_info.encode_type = request.encode_type();
        request_info.front_id = request.frontend_id();
        request_info.end_signal = request.end_signal();
        request_info.recompute = request.recompute();
        // std::string reply_info;
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        backend_->Exec(request_info);
    }
}

class ElasticcdcServiceImpl final : public ElasticcdcService::Service {
public:
    ElasticcdcServiceImpl(const std::string conf_path) : ElasticcdcService::Service() {
        LOG_INFO("init ElasticcdcService done");
    }

    Status DataTransStream(ServerContext* context, grpcStream* stream) override {
        
        // std::string prefix("ImageClassify ");
        ElasticcdcRequest request;
        while(stream->Read(&request) ) 
        {
            LOG_INFO("ElasticcdcService receive rpc DataTransStream, id: %ld, filename: %s, scale: %s, modle name: %s, data size: %ld",
                         request.id(), request.filename().c_str(), request.scale().c_str(), request.model_name().c_str(), request.data().size());
            std::lock_guard<std::mutex> lock(queueMutex);
            requestQueue.emplace(request, stream);

            queueCV.notify_one();
        }
        
        // need to wait for the procession completed
        return Status::OK;
    }

};

void RunServer(const std::string conf_path) {
    std::string server_address("0.0.0.0:50051");
    ElasticcdcServiceImpl service(conf_path);
  
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // start a processRequest thread
    std::thread processor(processRequests, conf_path);
  
    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
    processor.join();
}

void usage() {
    std::cout << "Usage: ./backend_server conf_path" << std::endl;
}

int main(int argc, char** argv) {
    // ImageClassify(argc, argv);
    if (argc != 2) {
        usage();
        return 0;
    }
    std::string conf_path;
    conf_path = std::string(argv[1]);

    RunServer(conf_path);
    return 0;
}