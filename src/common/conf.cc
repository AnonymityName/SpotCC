#include "conf.hh"
#include "logger.hh"

Config::Config(const std::string& _conf_path):conf_path(_conf_path) {}

Json::Value readJsonFile(const std::string & filename)
{
    std::ifstream ifile;
    ifile.open(filename);

    if(!ifile.is_open()) {
        std::cerr << "Fail to open file " << filename.c_str() << std::endl;  
    }

    
    Json::CharReaderBuilder ReaderBuilder;
    ReaderBuilder["emitUTF8"] = true;
    
    Json::Value root;
    
    std::string strerr;
    bool ok = Json::parseFromStream(ReaderBuilder, ifile, &root, &strerr);
    if(!ok){
        std::cerr << "Fail to parse json" << std::endl;
        exit(-1);
    }

    ifile.close();
    return root;
}

void Config::parse() {
    Json::Value root;
    root = readJsonFile(conf_path);
    try {
        node_number = root.get("node_number", 1).asUInt();
        LOG_INFO("Parsed node number: %d", node_number);

        frontend_id = root.get("frontend_id", 0).asUInt();
        LOG_INFO("Parsed frontend id: %d", frontend_id);

        test_mode = root.get("test_mode", 0).asString();
        LOG_INFO("Parsed test mode: %s", test_mode.c_str());

        // encoder_type = root.get("encoder", "null").asString();
        // decoder_type = root.get("decoder", "null").asString();
        dataset = root.get("dataset", "null").asString();
        model_ckpt = root.get("model_ckpt", "null").asString();
        // decoder_model = root.get("decoder_model", "null").asString();
        // decoder_ckpt = root.get("decoder_ckpt", "null").asString();
        output_path = root.get("output_path", "null").asString();
        cache_config = root["cache_config"];
        // batch_config = root["batch_config"];
        filter_config = root["filter_config"];
        preempted_check_interval = root.get("preempted_check_interval", 1).asUInt();
        use_cuda = root.get("use_cuda", false).asBool();
        LOG_INFO("Parsed preempted check interval: %d", preempted_check_interval);

        // if (use_cuda && torch::cuda::is_available()) {
        //     LOG_INFO("use_cuda is true, and cuda is available!");
        // }
        // else if (use_cuda && !torch::cuda::is_available()) {
        //     LOG_ERROR("use_cuda is true, but cuda is not available!");
        //     exit(1);
        // }
        // else {
        //     LOG_INFO("use_cuda is false!");
        // }
        
        encode_config = root.get("encode_config","null");
        if(!encode_config.isString()) {
            backup_num = encode_config.get("backup_num", 1).asUInt();
            LOG_INFO("Parsed backup size: %d", backup_num);

            encoder_type = encode_config.get("encoder_type", "null").asString();
            LOG_INFO("Parsed encoder type: %s", encoder_type.c_str());

            k = encode_config.get("k", "0").asUInt();
            LOG_INFO("Parsed encode parameter k: %d", k);
        } else {
            LOG_ERROR("Not find encode config!");
        }

        decode_config = root.get("decode_config", "null");
        if(!decode_config.isString()) {
            decoder_type = decode_config.get("decoder_type", "null").asString();
            LOG_INFO("Parsed decoder type: %s", decoder_type.c_str());

            decoder_ckpt = decode_config.get("decoder_ckpt", "null").asString();
            LOG_INFO("Parsed decoder ckpt: %s", decoder_ckpt.c_str());

            decoder_simulate = decode_config.get("decoder_simulate", false).asBool();
            LOG_INFO("Parsed decoder simulation: %d", decoder_simulate);
        } else {
            LOG_ERROR("Not find decode config!");
        }

        triton_config =  root.get("triton_config","null");
        if(!triton_config.isString()) {
            scale = triton_config.get("scale", "null").asString();
            model_name = triton_config.get("model", "null").asString();
            LOG_INFO("Parsed scale: %s, model name: %s", scale.c_str(), model_name.c_str());
        }else {
            LOG_ERROR("Not find triton config!");
        }

        preprocess_config = root.get("preprocess_config", "null");
        if(!preprocess_config.isString()) {
            format = preprocess_config.get("format", "null").asString();
            dtype = preprocess_config.get("dtype", "null").asString();
            channels = preprocess_config.get("channel", 3).asUInt();
            height = preprocess_config.get("height", 500).asUInt();
            width = preprocess_config.get("width", 300).asUInt();
            LOG_INFO("Parsed format: %s, dtype: %s, channels: %d, height: %d, width: %d",
                     format.c_str(), dtype.c_str(), channels, height, width);
        }else {
            LOG_ERROR("Not find preprocess config!");
        }

        filter_config = root.get("filter_config", "null");
        if (!filter_config.isString()) {
            filter_type = filter_config.get("type", "auto").asString();
            if (filter_type == "manual")
                cdc_ratio = filter_config.get("cdc_ratio", "0").asUInt();
        }
        
        auto regionArray = root.get("backend_IPs", Json::Value());
        if (regionArray.isArray()) {
            for (Json::Value::ArrayIndex i = 0; i < regionArray.size(); ++i) {
                
                auto trace_file_path = regionArray[i].get("trace","null").asString();
                if(trace_file_path == "null") {
                    LOG_ERROR("Not find trace file in region %d!", i);
                }

                auto start_time_slice = regionArray[i].get("start_time_slice", 0).asUInt();

                auto region_id = regionArray[i].get("region_id", 0).asUInt();

                auto ip_list_a_region = regionArray[i].get("ip_list", Json::Value());
                std::vector<std::string> ip_list{};
                if(ip_list_a_region.isArray()) {
                    for (Json::Value::ArrayIndex j = 0; j < ip_list_a_region.size(); ++j){
                        if (ip_list_a_region[j].isString()) {
                            ip_list.emplace_back(ip_list_a_region[j].asString());
                        }
                    }
                } else {
                    LOG_ERROR("Not find backend IPs in region %d!", i);
                }

                backend_IPs.emplace_back(std::make_tuple(trace_file_path, start_time_slice, region_id, ip_list));
                LOG_INFO("Parsed region %d current time slice:%d", i, start_time_slice);
            }
            
        } else {
            LOG_ERROR("Not find backend IPs!");
        }

        auto frontend_ips_array = root.get("frontend_ips", Json::Value());
        if (frontend_ips_array.isArray()) {
            for (Json::Value::ArrayIndex i = 0; i < frontend_ips_array.size(); ++i) {
                if (frontend_ips_array[i].isString()) {
                    frontend_ips.emplace_back(frontend_ips_array[i].asString());
                }
            }
        } else {
            LOG_ERROR("Not find backend IPs!");
        }

        batch_config = root.get("batch_config", "null");
        if (!batch_config.isString()) {
            batch_mode = batch_config.get("mode", "auto").asString();
            if (batch_mode == "manual") {
                if(batch_config.isMember("batch_size")) {
                    batch_size_1 = batch_config.get("batch_size", 1).asUInt();
                    batch_size_2 = batch_config.get("batch_size", 1).asUInt();
                }
                else {
                    batch_size_1 = batch_config.get("batch_size_1", 1).asUInt();
                    batch_size_2 = batch_config.get("batch_size_2", 1).asUInt();
                }
            }
            else if (batch_mode == "auto") {
                max_batch_size = batch_config.get("max_batch_size", 64).asUInt();
                inc_value = batch_config.get("inc_value", 10).asUInt();
                dec_value = batch_config.get("dec_value", 0.1).asDouble();
                batch_size_1 = max_batch_size/2;
                batch_size_2 = max_batch_size/2;
            }
            LOG_INFO("Parsed batch size, batch_size_1: %d, batch_size_2: %d", batch_size_1, batch_size_2);
        }
        else {
            LOG_ERROR("Not find batch config!");
        }

        client_config = root.get("client_config", "null");
        if (!client_config.isString()) {
            query_rate = client_config.get("query_rate", 1.0).asDouble();
            query_arrival_distribution = client_config.get("query_arrival_distribution", "auto").asString();
            workload_path = client_config.get("workload_path", "").asString();
            LOG_INFO("Parsed query_rate: %lf, query_arrival_distribution: %s", query_rate, query_arrival_distribution.c_str());
        }
        else {
            LOG_ERROR("Not find client config!");
        }

        monitor_config = root.get("monitor_config", "null");
        if (!monitor_config.isString()) {
            update_interval = monitor_config.get("update_interval", 10).asUInt();
            flag_algorithm = monitor_config.get("algorithm", "baseline").asString();
            recovery_time = monitor_config.get("recovery_time", 5).asUInt();
            cee = monitor_config.get("cee", false).asBool();
            to_vul_num = monitor_config.get("to_vul_num", 5).asUInt();
            rl_enchance = monitor_config.get("rl_enhance", false).asBool();
            update_mode = monitor_config.get("update_mode", "query").asString();
            alpha_decrease = monitor_config.get("alpha_decrease", 1.0).asDouble();
            alpha_increase = monitor_config.get("alpha_increase", 1.5).asDouble();
            history_length = monitor_config.get("history_length", 0).asUInt();
            top_k = monitor_config.get("top_k", 0).asUInt();
            eta_1 = monitor_config.get("eta_1", 0.1).asDouble();
            eta_2 = monitor_config.get("eta_2", 0.1).asDouble();
            if(update_mode == "time") {
                update_time_gap = monitor_config.get("update_time_gap", "query").asDouble();
                LOG_INFO("Parsed update_time_gap:%lf", update_time_gap);
            }
            LOG_INFO("Parsed update_interval: %d, flag_algorithm: %s, recovery_time: %d, to_vul_num: %d, update_mode: %s, alpha_decrease: %lf, alpha_increase: %lf",
                        update_interval, flag_algorithm.c_str(), recovery_time, to_vul_num, update_mode.c_str(), alpha_decrease, alpha_increase);
        }
        else {
            LOG_ERROR("Not find monitor config!");
        }
        
        arima_config = root.get("arima_config", "null");
        if (!arima_config.isString()) {
            arima_max_p = arima_config.get("max_p", 12).asUInt();
            arima_max_d = arima_config.get("max_d", 2).asUInt();
            arima_max_q = arima_config.get("max_q", 2).asUInt();
            LOG_INFO("Parsed arima_max_p: %d, arima_max_d: %d, arima_max_q: %d",
                        arima_max_p, arima_max_d, arima_max_q);
        }
    }
    catch (const Json::LogicError& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}
