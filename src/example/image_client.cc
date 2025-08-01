#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <grpcpp/grpcpp.h>


#include "../protocol/elasticcdc.grpc.pb.h"
#include <filesystem>
#include <chrono>
#include <future>

#include "../inc/inc.hh"
#include "../common/logger.hh"
#include "../common/conf.hh"


ABSL_FLAG(std::string, target, "localhost:50052", "Server address");


namespace fs = std::filesystem;

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReaderWriter;

typedef struct Latency {
    int id;
    double latency;
    Latency(int id, double latency):id(id), latency(latency) {}
}Latency;

bool compareByTime(const Latency& a, const Latency& b) {
    return a.latency < b.latency;
}

std::unordered_map<uint32_t, std::chrono::steady_clock::time_point> latency_map;
std::vector<double> latency_list;
std::vector<Latency> latency_struct_list;

/**
 * @brief calculate percetile latency
 */
double calculatePercentile(const std::vector<double>& latencyData, double percentile) {
    int index = static_cast<int>(latencyData.size() * percentile / 100.0);
    if (index == latencyData.size() * percentile / 100.0) {
        return latencyData[index - 1];
    }
    double fraction = latencyData.size() * percentile / 100.0 - index;
    return (1 - fraction) * latencyData[index - 1] + fraction * latencyData[index];
}

/**
 * @brief calculate average latency
 */
double calculateAverage(const std::vector<double>& latencyData) {
    double all = 0;
    for (const auto& data: latencyData) {
        all += data;
    }
    return all / (double)latencyData.size();
}

void vectorToCSV(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "latency" << std::endl;
        for (size_t i = 0; i < vec.size(); ++i) {
            file << vec[i];
            if (i < vec.size() - 1) {
                file << "," << std::endl;
            }
        }
        // file << std::endl;
        file.close();
        std::cout << "finish to write: " << filename << std::endl;
    } else {
        std::cerr << "cannot open: " << filename << std::endl;
    }
}

std::vector<double> readLatencyFromCSV(const std::string& filename) {
    std::vector<double> latencies;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open: " + filename);
    }
    
    std::string line;
    
    if (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        if (line != "latency") {
            throw std::runtime_error("CSV file wrong format, expected title is 'latency'");
        }
    } else {
        throw std::runtime_error("file is empty!");
    }
    
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        
        if (line.back() == '\r') {
            line.pop_back();
        }
        
        try {
            double latency = std::stod(line);
            latencies.push_back(latency);
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error("Invalid format: " + line + " - " + e.what());
        } catch (const std::out_of_range& e) {
            throw std::runtime_error("data out of range: " + line + " - " + e.what());
        }
    }
    
    if (file.bad()) {
        throw std::runtime_error("Fail to read file!");
    }
    
    return latencies;
}


/**
 * @brief Distribution interface
 */
class IntervalDistribution {
    public:
        virtual double generateInterval() = 0;
        virtual ~IntervalDistribution() {}
};

/**
 * @brief Poisson distribution
 */
class PoissonDistribution : public IntervalDistribution {
public:
    PoissonDistribution(double queryRate): rd(), gen(rd()), expDist(queryRate) {}
    double generateInterval() override {
        return expDist(gen);
    }
private:
    std::random_device rd;
    std::mt19937 gen;
    std::exponential_distribution<> expDist;
};

class BurstyDistribution:public IntervalDistribution {
public:
    BurstyDistribution() {}
    double generateInterval() override {
        return 0;
    }
};

class MAFDistribution: public IntervalDistribution {
public:
    MAFDistribution(std::string distPath) {
        index_ = 0;
        intervalList_ = readLatencyFromCSV(distPath);
        listLen_ = intervalList_.size();
    }
    double generateInterval() override {
        return intervalList_[(index_++) % listLen_];
    }
private:
    std::vector<double> intervalList_;
    size_t index_;
    size_t listLen_;
};


/**
 * @brief read image files
 */
std::string ReadImageFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filename.c_str());
        return "";
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

/**
 * @brief send Image
 */
void SendImages(std::vector<std::shared_ptr<grpcStreamClient>> streams,
                 const std::string& directory,
                 const std::shared_ptr<Config>& conf,
                std::promise<int>& image_num,
                std::promise<int>& sleep_time)
{
    int _id = 0; // image id
    int cnt = 0;
    int sleep = 0;

    std::shared_ptr<IntervalDistribution> distribution;
    if (conf->query_arrival_distribution == "poisson") {
        distribution = std::make_shared<PoissonDistribution>(conf -> query_rate);
    } 
    else if (conf->query_arrival_distribution == "bursty") {
        distribution = std::make_shared<BurstyDistribution>();
    }
    else if (conf->query_arrival_distribution == "MAF") {
        distribution = std::make_shared<MAFDistribution>(conf->workload_path);
    }
    else {
        LOG_INFO("Not define the %s distribution ", conf->query_arrival_distribution.c_str());
        exit(1);
    }
    
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
            cnt++;
            std::string image_data = ReadImageFile(entry.path().string());

            if (!image_data.empty()) {
                std::string filename = entry.path().filename();
                ElasticcdcRequest request;
                // fill the request info
                request.set_data(image_data);
                request.set_id(_id++);
                request.set_model_name(conf->model_name);
                request.set_scale(conf->scale);
                request.set_filename(filename);
                request.set_end_signal(false);

                latency_map[_id-1] = std::chrono::steady_clock::now();
                LOG_INFO("Client sending image %s with size %ld bytes, id: %d", filename.c_str(), image_data.size(), _id - 1);
                
                static std::random_device rd;  
                static std::mt19937 gen(rd()); 

                std::uniform_int_distribution<> distrib(0, streams.size() - 1);

                int idx = distrib(gen);
                streams[idx]->Write(request);

                // Send data in a certain distribution.
                int intervalMs = static_cast<int>(distribution->generateInterval() * 1000);
                sleep += intervalMs;
                std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
            }
        }
    }
    // send end signal
    ElasticcdcRequest request;
    request.set_id(0);
    request.set_end_signal(true);
    LOG_INFO("Client sending end signal!");
    // for (int i = 0; i < streams.size(); i++)
    //     streams[i]->Write(request);
    streams[0]->Write(request);
    for (int i = 0; i < streams.size(); i++)
        streams[i]->WritesDone();
    image_num.set_value(cnt);
    sleep_time.set_value(sleep);
}

void ReceiveImages(std::shared_ptr<grpcStreamClient> stream) {
    ElasticcdcReply reply;
    int imageIndex = 0;
    while (stream->Read(&reply)) {
        LOG_INFO("Client received processed image %ld with size %ld bytes", reply.id(), reply.reply_info().size());
        // LOG_INFO("Reply info is: %s", reply.reply_info().c_str());
        imageIndex++;
        auto end = std::chrono::steady_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - latency_map[reply.id()]);
        if (!reply.recompute()) {
            latency_list.push_back(latency.count());
            latency_struct_list.push_back(Latency(reply.id(), latency.count()));
        }
            
    }

    LOG_INFO("Total %d images", imageIndex);
}

void RunClient(const std::string& conf_path, const std::string& directory) {
    auto conf = std::make_shared<Config>(conf_path);
    conf->parse();

    // connect to all frontends
    std::vector<std::pair<std::shared_ptr<Channel>, std::unique_ptr<ElasticcdcService::Stub>>> channel_stub_pairs;
    std::vector<std::shared_ptr<ClientContext>> contexts;
    std::vector<std::shared_ptr<grpcStreamClient>> streams;

    for (int i = 0; i < conf->frontend_ips.size(); i++) {
        std::string addr = conf->frontend_ips[i] + ":50052";
        auto channel = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
        auto stub = ElasticcdcService::NewStub(channel);
        channel_stub_pairs.emplace_back(channel, std::move(stub));

        auto context = std::make_shared<ClientContext>();
        auto stream = std::shared_ptr<grpcStreamClient>(
            channel_stub_pairs[i].second->DataTransStream(context.get())
        );

        contexts.push_back(context);
        streams.push_back(stream);
    }
    
    // create threads
    auto start = std::chrono::high_resolution_clock::now();
    std::promise<int> image_num;
    std::future<int> image_num_fut = image_num.get_future();
    std::promise<int> sleep_time;
    std::future<int> sleep_time_fut = sleep_time.get_future();
    std::thread sender(SendImages, std::ref(streams), directory, conf, std::ref(image_num), std::ref(sleep_time));
    std::vector<std::shared_ptr<std::thread>> receivers;
    for (int i = 0; i < conf->frontend_ips.size(); i++) {
        auto receiver = std::make_shared<std::thread>(ReceiveImages, std::ref(streams[i]));
        receivers.push_back(receiver);
    }

    // SendImages(stream, directory, conf);
    // ReceiveImages(stream);

    // wait
    sender.join();
    for (int i = 0; i < conf->frontend_ips.size(); i++) {
        receivers[i]->join();
    }
    
    for (int i = 0; i < conf->frontend_ips.size(); i++) {
        Status status = streams[i]->Finish();
        if (status.ok()) {
            LOG_INFO("RPC completed successfully");
        } else {
            LOG_ERROR("RPC failed: %s",status.error_message().c_str());
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int image_num_int = image_num_fut.get();
    int sleep_time_int = sleep_time_fut.get();

    auto duration_long = static_cast<long>(duration);
    // duration_long -= static_cast<long>(sleep_time_int);

    LOG_INFO("Sleep time: %d ms", sleep_time_int);

    LOG_INFO("End-to-end latency: %ld ms", duration_long);

    // double ave_latency = static_cast<double>(duration_long) / static_cast<double>(image_num_int);

    // LOG_INFO("Image number: %d, Average latency: %lf ms", image_num_int, ave_latency);

    double throughput = static_cast<double>(image_num_int) * 1000 / static_cast<double>(duration_long);

    LOG_INFO("Throughput: %lf queries/s", throughput);

    assert(latency_list.size() == image_num_int);

    sort(latency_list.begin(), latency_list.end());
    for (int i = 0; i < conf->k; i++) latency_list.pop_back();
    // sort(latency_struct_list.begin(), latency_struct_list.end(), compareByTime);

    double p50 = calculatePercentile(latency_list, 50);
    double p90 = calculatePercentile(latency_list, 90);
    double p95 = calculatePercentile(latency_list, 95);
    double p99 = calculatePercentile(latency_list, 99);

    LOG_INFO("P50 latency: %lf ms", p50);
    LOG_INFO("P90 latency: %lf ms", p90);
    LOG_INFO("P95 latency: %lf ms", p95);
    LOG_INFO("P99 latency: %lf ms", p99);

    // int id = (int)(0.99 * (double)latency_struct_list.size() / (double) 100);
    // LOG_INFO("P99 id: %d", latency_struct_list[id].id);

    double average = calculateAverage(latency_list);

    LOG_INFO("Average latency: %lf ms", average);

    vectorToCSV(latency_list, "./latency.csv");
}

void usage() {
  std::cout << "Usage: ./image_client conf_path data_path" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        usage();
        return 0;
    }
    
    std::string conf_path = argv[1];
    std::string directory = argv[2];
    RunClient(conf_path, directory);
    return 0;
}