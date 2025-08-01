#pragma once

#include "../inc/inc.hh"
#include "zone.hh"
#include "../util/json/json.h"
#include "../common/conf.hh"
#include "../util/ARIMA/ARIMA.hh"

extern Json::Value readJsonFile(const std::string & filename);

struct TraceInfo{
    Json::Value trace_data;
    uint32_t cur_time_slice_{0};
    uint32_t start_time_slice_{0};
    std::vector<std::string> ip_list;

    TraceInfo(const Json::Value& data, uint32_t cur_time_slice,const std::vector<std::string>& list)
        : trace_data(data), cur_time_slice_(cur_time_slice), start_time_slice_(cur_time_slice), ip_list(list) {}
};

class Monitor {

private:
    // uint32_t cur_time_slice_{0};

    std::vector<TraceInfo> traces_info_{};  // the available nodes num and the ips list of a zone
    // std::vector<std::string> backend_ips_{};

    std::thread monitor_update_thread_;
    std::mutex mtx_;
    double update_time_gap;
    bool isfinish = false;
    int k_;

    std::shared_ptr<std::mutex> mtx_1_;
    std::shared_ptr<std::condition_variable> cv_1_;
    bool* is_notify_;

    std::vector<ZoneState> zones_state_{};
    std::vector<std::string> ava_invul_backend_ips_{};
    std::vector<std::string> ava_vul_backend_ips_{};
    std::vector<std::string> backend_ips_{};
    std::unordered_set<std::string> new_unava_backend_ips_{};

    std::string flag_algorithm_ = "passive";

    uint32_t arima_max_p;
    uint32_t arima_max_d;
    uint32_t arima_max_q;

    double alpha_d;
    double alpha_i;

    std::unordered_map<std::string, std::unordered_map<uint32_t, bool>> querys_state{}; //[backendip, [query_id, is_not_broken]]
    std::mutex mtx_querys_state_;

public:
    Monitor() = delete;
    Monitor(std::shared_ptr<Config> conf_,
            std::shared_ptr<std::mutex> mtx_1,
            std::shared_ptr<std::condition_variable> cv_1,
            bool* is_notify):
            mtx_1_(mtx_1),
            cv_1_(cv_1),
            is_notify_(is_notify) { 

        for(auto region_trace: conf_ -> backend_IPs) {
            auto [trace_file_path, start_time_slice, ip_list] = region_trace;
            auto trace_data = readJsonFile(trace_file_path).get("data", Json::Value());
            if(trace_data.isArray()) {
                traces_info_.emplace_back(trace_data, start_time_slice, ip_list);
            }
            backend_ips_.insert(backend_ips_.end(), ip_list.begin(), ip_list.end());
            // initialize the zones and nodes
            zones_state_.emplace_back(ip_list.size(), conf_->recovery_time, conf_->to_vul_num);
        }
        flag_algorithm_ = conf_ -> flag_algorithm;
        update_time_gap = conf_ -> update_time_gap;

        arima_max_p = conf_->arima_max_p;
        arima_max_d = conf_->arima_max_d;
        arima_max_q = conf_->arima_max_q;

        alpha_d = conf_->alpha_decrease;
        alpha_i = conf_->alpha_increase;

        k_ = conf_->k;
        querys_state.reserve(backend_ips_.size());

        if(flag_algorithm_!="baseline") {
            FlagZoneInitialization();
        }

        // findStartTime();
        SelectAvaBackendIPs();

        // depand on the mode of monitor update(in conf file)
        if(conf_->update_mode=="query") 
            monitor_update_thread_ = std::thread(&Monitor::Update, this);
        else if(conf_->update_mode=="time")
            monitor_update_thread_ = std::thread(&Monitor::UpdateWithTime, this);

        LOG_INFO("Monitor created.");
    }

    ~Monitor() {
        monitor_update_thread_.join();
    }

    void Update() {
        while(true) {
            {
                std::unique_lock<std::mutex> lock(*mtx_1_);
                LOG_INFO("Monitor wait for update...");
                cv_1_->wait(lock, [this] {return *is_notify_;});
                *is_notify_ = false;
            }
            // cur_time_slice_++;
            // assert(cur_time_slice_ < traces_info_.front().trace_data.size());

            updateZonesState();
            // std::cout << 1 << std::endl;

            if(flag_algorithm_!="baseline")
                FlagZones();
            // std::cout << 2 << std::endl;
            std::lock_guard<std::mutex> lock(mtx_);
            SelectAvaBackendIPs();
            UpdateQuerysState();

            // if(flag_algorithm_!="baseline" && ava_invul_backend_ips_.size() == backend_ips_.size()) {
            //     FlagZoneInitialization();
            //     SelectAvaBackendIPs();
            // }

            if (flag_algorithm_ != "baseline") {
                for (int i = 0; i < zones_state_.size(); i++) {
                    auto zone = zones_state_[i];
                    if (zone.get_available_nodes_id().size() > 0 && zone.get_invulnerable_nodes_id().size() > 0) {
                        if (zone.get_available_nodes_id().size() == zone.get_invulnerable_nodes_id().size()) {
                            FlagZoneInitialization(i);
                        }
                    }
                }
                SelectAvaBackendIPs();
            }
        }
    }


    void UpdateWithTime() {
        auto time_gap = static_cast<long long>(update_time_gap);
        while (true) {
            // LOG_INFO("Monitor wait for update...");
            std::this_thread::sleep_for(std::chrono::milliseconds(time_gap)); 
            
            updateZonesState();
            if(isfinish) break;

            if(flag_algorithm_!="baseline")
                FlagZones();
            // std::cout << 2 << std::endl;
            std::lock_guard<std::mutex> lock(mtx_);
            SelectAvaBackendIPs();

            UpdateQuerysState();

            if(flag_algorithm_!="baseline" && ava_invul_backend_ips_.size() == backend_ips_.size()) {
                FlagZoneInitialization();
                SelectAvaBackendIPs();
            }
        }
    }

     /*
     * @brief judge whether current time slice of region i is in the frequent period or not
     * @note use algorithm to find the frequent period at the beginning or use machine learning alg.
     */
    auto isInFreqPeriod(uint32_t cur_time_slice_, uint32_t region_id) -> bool {
        return false;
    }

    auto updateZonesState() -> void {
        for(int i = 0; i < zones_state_.size(); i++) {
            auto& info = traces_info_[i];
            info.cur_time_slice_++;
            LOG_INFO("Update zone %d, current time slice:%d", i, info.cur_time_slice_);
            if(info.cur_time_slice_ >= info.trace_data.size()) {
                isfinish = true;
                break;
            }

            zones_state_[i].updateZoneEachTime(info.trace_data[info.cur_time_slice_].asUInt());
        }
        // LOG_INFO("Update zones state completed!");
    }

    /*
     * @brief select backend ips of all regions, and devide them into invulnerable and vulnerable
     * @note update ava_invul_backend_ips_ and ava_vul_backend_ips_
     * @note std::move do not need copy
     */
    auto SelectAvaBackendIPs() -> void {
        LOG_INFO("Select available backend ips start!");
        std::vector<std::string> ava_invul_ips{};
        std::vector<std::string> ava_vul_ips{};
        std::unordered_set<std::string> new_unava_ips{};
        for(int i = 0; i < traces_info_.size(); i++) {
            auto ip_list_a_region = SelectAvaBackendIPsARegion(i);
            std::move(std::get<0>(ip_list_a_region).begin(), std::get<0>(ip_list_a_region).end(),
                                                         std::back_inserter(ava_invul_ips));
            std::move(std::get<1>(ip_list_a_region).begin(), std::get<1>(ip_list_a_region).end(),
                                                         std::back_inserter(ava_vul_ips));
            new_unava_ips.insert(std::get<2>(ip_list_a_region).begin(), std::get<2>(ip_list_a_region).end());
        }

        // lock
        ava_invul_backend_ips_ = ava_invul_ips;
        ava_vul_backend_ips_ = ava_vul_ips;
        new_unava_backend_ips_ = new_unava_ips;
        LOG_INFO("Select available backend ips completed!");
        return;
    }

    auto SelectAvaBackendIPsARegion(uint32_t region_id) const -> std::tuple<std::vector<std::string>,
                                                                 std::vector<std::string>, std::unordered_set<std::string>> {
        assert(region_id >= 0 && region_id < traces_info_.size());
        LOG_INFO("SelectAvaBackendIPsARegion start!");
        std::vector<std::string> ava_invul_ips{};
        std::vector<std::string> ava_vul_ips{};
        std::unordered_set<std::string> new_unava_ips{};

        auto trace_info = traces_info_[region_id];
        auto ip_list = trace_info.ip_list;
        auto ava_nodes_id = zones_state_[region_id].get_available_nodes_id();
        auto invul_nodes_id = zones_state_[region_id].get_invulnerable_nodes_id();
        auto vul_nodes_id = zones_state_[region_id].get_vulnerable_nodes_id();
        auto new_unava_nodes_id = zones_state_[region_id].get_new_unavailable_nodes_id();

        for(const auto& index: ava_nodes_id){
            if(invul_nodes_id.find(index) != invul_nodes_id.end()) 
                ava_invul_ips.emplace_back(ip_list[index]);
            else   
                ava_vul_ips.emplace_back(ip_list[index]);
        }
        for(const auto& index: new_unava_nodes_id){
            new_unava_ips.insert(ip_list[index]);
        }
        // std::cout << "ava nodes num:" << ava_nodes_id.size() << std::endl;
        // std::cout << "available invulnerable size:" << ava_invul_ips.size() << std::endl;
        // std::cout << "available vulnerable size:" << ava_vul_ips.size() << std::endl;

        std::cout << "region id " << region_id << std::endl;
        std::cout << "ava nodes num:" << ava_nodes_id.size() << std::endl;
        std::cout << "available invulnerable size:" << ava_invul_ips.size() << std::endl;
        for(const auto& ip:ava_invul_ips) {
            std::cout << ip << std::endl;
        }
        std::cout << "available vulnerable size:" << ava_vul_ips.size() << std::endl;
        for(const auto& ip:ava_vul_ips) {
            std::cout << ip << std::endl;
        }
        std::cout << "new preempted nodes size:" << new_unava_ips.size() << std::endl;
        for(const auto& ip:new_unava_ips) {
            std::cout << ip << std::endl;
        }


        return std::make_tuple(ava_invul_ips, ava_vul_ips, new_unava_ips);
    }

    auto FlagZonePassive(uint32_t region_id) -> void {
        zones_state_[region_id].updateFlagPassive();
    }

    auto FlagZoneActive(uint32_t region_id) -> void {
        auto pred_available_nodes_num = get_pred_ava_nodes_num(region_id);
        zones_state_[region_id].updateFlagPred(pred_available_nodes_num);
    }

    auto FlagZones() -> void {
        for(int i = 0; i < zones_state_.size(); i++) {
            auto info = traces_info_[i];
            auto cur_time_slice = info.cur_time_slice_;
            auto start_time_slice = info.start_time_slice_;

            if(flag_algorithm_ == "auto") 
                flag_algorithm_ = isInFreqPeriod(cur_time_slice, i) ? "active" : "passive";

            if(flag_algorithm_ == "passive" || cur_time_slice - start_time_slice < arima_max_p*2) 
                FlagZonePassive(i);
            else if(flag_algorithm_ == "active" && cur_time_slice - start_time_slice >= arima_max_p*2) // need to rethink!!!!
                FlagZoneActive(i);
            else {
                LOG_ERROR("Flag algorithm %s error", flag_algorithm_.c_str());
                exit(-1);
            }
            // LOG_INFO("Flag zone %d, current time slice:%d", i, cur_time_slice);
        }

        
    }

    auto FlagZoneInitialization() -> void {
        for(int i = 0; i < zones_state_.size(); i++) {
            zones_state_[i].FlagInitialization();
        }
        LOG_INFO("Flag Zone Initialization completed!");
        
    }

    auto FlagZoneInitialization(int zone_id) -> void {
        zones_state_[zone_id].FlagInitialization();
        LOG_INFO("Flag Zone %d Initialization completed!", zone_id);
    }

    auto get_pred_ava_nodes_num(int region_id) const -> uint32_t{
        // LOG_INFO("Not implement get_pred_ava_nodes_num function");

        std::vector<double> data;

        auto info = traces_info_[region_id];
        auto cur_time_slice = info.cur_time_slice_;
        auto start_time_slice = info.start_time_slice_;
        assert(cur_time_slice - start_time_slice >= arima_max_p*2);

        for (int i = arima_max_p*2-1; i >= 0; i--) {
            data.push_back(info.trace_data[cur_time_slice - i].asDouble());
        }
        auto start = std::chrono::high_resolution_clock::now();
        ARIMA model = select_best_model(data, arima_max_p, arima_max_d, arima_max_q);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // std::cout << "Training time: " << duration << " us" << std::endl;
    
        double next = model.predict(data);
        // std::cout << "Next prediction: " << next << std::endl;
    
        std::vector<double> forecast = model.forecast(data, 1);

        return forecast[0];
    }

    auto get_ava_invul_backendIPS() -> std::vector<std::string> {
        std::lock_guard<std::mutex> lock(mtx_);
        // std::cout << "available invulnerable size:" << ava_invul_backend_ips_.size() << std::endl;
        return ava_invul_backend_ips_;
    }

    auto get_ava_vul_backendIPS() -> std::vector<std::string> {
        std::lock_guard<std::mutex> lock(mtx_);
        // std::cout << "available vulnerable size:" << ava_vul_backend_ips_.size() << std::endl;
        return ava_vul_backend_ips_;
    } 

    auto get_ava_backendIPs() -> std::vector<std::string> {
        std::lock_guard<std::mutex> lock(mtx_);
        std::vector<std::string> result;

        result.insert(result.end(),ava_invul_backend_ips_.begin(),ava_invul_backend_ips_.end());
        result.insert(result.end(),ava_vul_backend_ips_.begin(),ava_vul_backend_ips_.end());

        return result;
    }

    auto get_all_backendIPs() -> std::vector<std::string> {return backend_ips_;}

    auto findAvaBackendIPs() -> bool {
        std::lock_guard<std::mutex> lock(mtx_);
        if(ava_invul_backend_ips_.size() + ava_vul_backend_ips_.size() <= k_) return false;
        return true;
    }


    auto sendQueryToABackend(std::string backendip, uint32_t query_id) -> void {
        // std::cout << "send query "<< query_id << " to a backend " << backendip << " start!";
        std::lock_guard<std::mutex> lock(mtx_querys_state_);
        querys_state[backendip][query_id] = true;
        // std::cout << "send query "<< query_id << " to a backend " << backendip << " completed!";
    }

    /*
     *@brief update querys state after
     */
    auto UpdateQuerysState() -> void {
        std::lock_guard<std::mutex> lock(mtx_querys_state_);
        for(const auto& ip: new_unava_backend_ips_) {
            std::cout << "new unava ip:" << ip << std::endl;
            if (querys_state.find(ip) != querys_state.end()) {
                for(auto& query_state: querys_state[ip]) {
                    if (query_state.second) {
                        std::cout << "set query id: " << query_state.first << " as false" << std::endl;
                    }
                    query_state.second = false;
                }
            }
        }

    }

    auto IsQueryBroken(std::string backendip, uint32_t query_id) -> bool {
        std::lock_guard<std::mutex> lock(mtx_querys_state_);
        return !querys_state[backendip][query_id];
    }

    auto DeleteAQueryState(std::string backendip, uint32_t query_id) -> void {
        std::lock_guard<std::mutex> lock(mtx_querys_state_);
        querys_state[backendip].erase(query_id);
    }


    /*
     *@brief:choose a backend to send a task
    */
    auto ChooseABackend(bool is_parity_data) -> std::string {
        std::lock_guard<std::mutex> lock(mtx_);
        // choose a zone 
        auto chosen_zone_id = ChooseAZoneRandomly(is_parity_data);
        auto zone = zones_state_[chosen_zone_id];
        
        // choose a backend from the zone
        auto trace_info = traces_info_[chosen_zone_id];
        auto ip_list = trace_info.ip_list;
        uint32_t node_id = 0;
        if(!zone.get_available_nodes_num()) {
            node_id = zone.get_node_id();
        } else if(flag_algorithm_ == "baseline") {
            node_id = zone.get_ava_node_id();
        } else if(is_parity_data) {
            node_id = zone.get_ava_node_id_decrease(alpha_d);
        }
        else node_id = zone.get_ava_node_id_increase(alpha_i); 

        auto chosen_backend_ip = ip_list[node_id];
        LOG_INFO("Choose backend %s in zone %d.", chosen_backend_ip.c_str(), chosen_zone_id);
        return chosen_backend_ip;
    }

    /*
     *@brief:choose a zone to send task
     *@node:choose a zone randomly (if the task is not the parity task, choose the zone that the nodes num is bigger than one),
            anyway the zone chosen has a available node at least. If there is no available zone, return a node of all zone randomly.
    */
    auto ChooseAZoneRandomly(bool is_parity_data) const -> uint32_t {
        std::vector<int> ava_zone_id_1;   // only have a node in the zone
        std::vector<int> ava_zone_id;     // have more than a node in the zone
        ava_zone_id_1.reserve(zones_state_.size());
        ava_zone_id.reserve(zones_state_.size());

        for(int i = 0; i < zones_state_.size(); i++) {
            auto ava_nodes_num = zones_state_[i].get_available_nodes_num();
            if(ava_nodes_num == 1) ava_zone_id_1.emplace_back(i);
            else if(ava_nodes_num > 1) ava_zone_id.emplace_back(i);
        }

        if(is_parity_data || flag_algorithm_ == "baseline") 
            ava_zone_id.insert(ava_zone_id.end(),ava_zone_id_1.begin(), ava_zone_id_1.end());


        std::random_device rd;
        std::mt19937 gen(rd());
        if(!ava_zone_id.size()) {
            std::uniform_int_distribution<> dis(0, zones_state_.size() - 1);
            return dis(gen);
        }

        std::uniform_int_distribution<> dis(0, ava_zone_id.size() - 1);
        int randomIndex = dis(gen);
        return ava_zone_id[randomIndex];
    }
};
