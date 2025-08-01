#include <iostream>
#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>

#ifdef BAZEL_BUILD
#include "examples/protos/elasticcdc.grpc.pb.h"
#else
#include "../protocol/elasticcdc.grpc.pb.h"
#endif
#include "../common/logger.hh"
#include "../common/conf.hh"
#include "../util/json/json.h"
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using elasticcdc::ElasticcdcRequest;
using elasticcdc::ElasticcdcReply;
using elasticcdc::ElasticcdcService;


class MonitorClient {
  public:
    MonitorClient(std::shared_ptr<Channel> channel)
      : stub_(ElasticcdcService::NewStub(channel)) {
      conf_ = std::make_shared<Config>("../conf/1.json");
      conf_->parse();
    }

    void Run() {
      while (true) {
        sleep(conf_->getPreemptedCheckInterval());
        if (IsPreempted()) {
          LOG_INFO("rpc server is preempted");
        } else {
          LOG_INFO("rpc server is not preempted");
        }
      }
    }

private:
    bool IsPreempted() {
      ElasticcdcRequest request;
      ElasticcdcReply reply;
      ClientContext context;
      Status status;
      // Status status = stub_->IsPreempted(&context, request, &reply);
      if (status.ok()) {
        return false;   // not preempted
      } else {
        return true;    // is preempted
      }
    }
  
    std::unique_ptr<ElasticcdcService::Stub> stub_;
    std::shared_ptr<Config> conf_;
};

/**
 * send rpc call to rpc server to check whether server is preempted
 */
int main(int argc, char** argv) {
  const std::string target_str = "localhost:50051";
  MonitorClient client(grpc::CreateChannel(
                          target_str, grpc::InsecureChannelCredentials()));

  client.Run();
  return 0;
}



