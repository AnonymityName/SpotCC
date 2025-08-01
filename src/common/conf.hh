#pragma once

#include "../inc/inc.hh"
#include "../util/json/json.h"

class Config {
public:
    Config(const std::string& _conf_path);
    void parse();
    uint32_t getNodeNumgber() { return node_number;}
    Json::Value getCacheConfig() const { return cache_config; }
    Json::Value getBatchConfig() const { return batch_config; }
    uint32_t getPreemptedCheckInterval() { return preempted_check_interval; }

    std::string conf_path;
    uint32_t node_number;
    uint32_t k;
    std::string test_mode;
    
    std::string encoder_type;
    std::string decoder_type;
    std::string dataset;
    std::string model_ckpt;
    std::string decoder_model;
    std::string decoder_ckpt;
    bool decoder_simulate;
    std::string output_path;
    uint32_t preempted_check_interval;
    Json::Value cache_config;
    Json::Value batch_config;
    Json::Value encode_config;
    Json::Value decode_config;
    Json::Value triton_config;
    Json::Value preprocess_config;
    Json::Value filter_config;
    Json::Value client_config;
    Json::Value monitor_config;
    Json::Value arima_config;
    
    uint32_t backup_num;
    // std::vector<std::pair<std::string, std::vector<std::string>>> backend_IPs{};  //[region, [trace_file_path, ip_list]]
    std::vector<std::tuple<std::string, uint32_t, uint32_t, std::vector<std::string>>> backend_IPs{}; //[region, [trace_file_path, start_time_slice, ip_list]]

    std::string scale;
    std::string model_name;

    // preprocess config
    std::string format;
    std::string dtype;
    std::string filter_type;
    uint32_t cdc_ratio;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    bool use_cuda;
    std::string batch_mode;
    uint32_t batch_size_1;
    uint32_t batch_size_2;
    uint32_t default_batch_size;
    uint32_t max_batch_size;
    uint32_t inc_value;
    double dec_value;

    // client config
    double query_rate;
    std::string query_arrival_distribution;
    std::string workload_path;

    // monitor config
    uint32_t update_interval;
    std::string flag_algorithm;
    uint32_t recovery_time;
    bool cee;
    uint32_t to_vul_num;
    bool rl_enchance;
    std::string update_mode;
    double update_time_gap;
    double alpha_decrease;
    double alpha_increase;
    uint32_t history_length;
    uint32_t top_k;
    double eta_1;
    double eta_2;

    // arima config
    uint32_t arima_max_p;
    uint32_t arima_max_d;
    uint32_t arima_max_q;

    uint32_t frontend_id;
    std::vector<std::string> frontend_ips{};
};
