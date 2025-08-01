#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "../protocol/elasticcdc.grpc.pb.h"


#include "../inc/inc.hh"
#include "../common/image.hh"
#include "../common/logger.hh"
#include "../frontend/frontend.hh"
// #include "../frontend/monitor.hh"
#include "../frontend/monitor2parts.hh"
#include "../common/conf.hh"
// #include "../util/queue.hh"


using grpc::ServerReaderWriter;
using RequestQueue = std::queue<std::pair<ElasticcdcRequest, std::shared_ptr<grpcStream>>>;

RequestQueue requestQueue;
std::mutex queueMutex;
std::condition_variable queueCV;

std::mutex completedMutex;
std::condition_variable completedCV;

std::shared_ptr<std::mutex> taskCountMtx = std::make_shared<std::mutex>();
int tasks_completed_num = 0;
int start_task_id = 0;
int task_num = 0;

std::shared_ptr<std::mutex> monitorMtx;
std::shared_ptr<std::condition_variable> monitorCV;
bool is_notify_ = false;

void processRequests(std::shared_ptr<Config> conf, std::shared_ptr<Monitor> monitor, std::shared_ptr<Filter> filter) {
    auto frontend_ = std::make_shared<Frontend>(conf, monitor, filter);
    uint32_t query_num = 0;
    while(true) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCV.wait(lock, [] { return !requestQueue.empty(); });

        auto [request, stream] = requestQueue.front();
        requestQueue.pop();
        lock.unlock();

        // send reply
        ImageArgs request_info;
        request_info.filename = request.filename();
        request_info.model_name = request.model_name();
        request_info.id = request.id();
        request_info.scale = request.scale();
        request_info.data.assign(request.data().begin(),request.data().end());
        request_info.stream = stream;
        request_info.end_signal_ = request.end_signal();
        request_info.is_recompute_ = false;
        request_info.is_parity_data_ = false;
        // std::string reply_info;

        // notify monitor
        // std::cout << "query_num:" << query_num << " conf -> update_interval:" << conf -> update_interval << std::endl;
        // if(conf->update_mode == "query" && ! ((query_num + 1) % conf -> update_interval)) {
        //   std::cout << "query_num:" << query_num << std::endl;
        //   std::lock_guard<std::mutex> lock(*monitorMtx);
        //   is_notify_ = true;
        //   monitorCV->notify_one();
        // }
        // query_num++;

        frontend_->Exec(request_info);
    }
}

// Logic and data behind the server's behavior.
class DataTransServiceImpl final : public ElasticcdcService::Service  {

public:
  DataTransServiceImpl(const std::string conf_path) : ElasticcdcService::Service() {
    LOG_INFO("Init DataTransService done");
	}

  Status DataTransStream(ServerContext* context, grpcStream* stream) override {
    ElasticcdcRequest request;
    while(stream->Read(&request)) {
        int64_t id = request.id();
        int64_t new_id = id + start_task_id;
        LOG_INFO("DataTransService receive rpc DataTransStream, id: %ld to id: %ld" , id, new_id);
        request.set_id(new_id);
        if(!request.end_signal()) task_num++;
        // std::string prefix("ImageClassify ");

        std::shared_ptr<grpcStream> sharedStream(stream, [](grpcStream* ptr) {
          delete ptr;
        });
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            requestQueue.emplace(request, sharedStream);
        }
        queueCV.notify_one();
    }

    //wait for all the images processed
    {
      std::unique_lock<std::mutex> lock(completedMutex);
      completedCV.wait(lock, [&]{ return task_num == tasks_completed_num; });
      start_task_id = tasks_completed_num;
    }

    return Status::OK;
  }

};

void RunServer(const std::string& conf_path) {
  std::string server_address("0.0.0.0:50052");
  DataTransServiceImpl service(conf_path);

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
  LOG_INFO( "Server listening on %s", server_address.c_str());

  auto conf =  std::make_shared<Config>(conf_path);
  conf->parse();

  auto filter = std::make_shared<Filter>(conf);

  monitorMtx = std::make_shared<std::mutex>();
  monitorCV = std::make_shared<std::condition_variable>();
  auto monitor = std::make_shared<Monitor>(conf, filter, monitorMtx, monitorCV, &is_notify_);

  // start a processRequest thread
  std::thread processor(processRequests,conf,monitor,filter);

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
  processor.join();
}

void usage() {
  std::cout << "Usage: ./image_frontend conf_path" << std::endl;
}

int main(int argc, char** argv) {

  if (argc != 2) {
    usage();
    return 0;
  }

  std::string conf_path = argv[1];
  RunServer(conf_path);
  return 0;
}