#pragma once

#include "../inc/inc.hh"
#include "zone2parts.hh"
#include "../util/json/json.h"
#include "../common/conf.hh"
#include "../util/ARIMA/ARIMA.hh"
#include "filter.hh"
#include <deque>
#include <queue>
#include <cmath>
#include <map>
#include <utility>

extern Json::Value readJsonFile(const std::string & filename);

using State = std::pair<double, double>;

class RLAgent {
private:
    std::vector<State> stateSpace;
    std::vector<double> actionSpace;
    
    std::map<State, std::map<double, double>> qTable;
    
    double learningRate;
    double discountFactor;
    double explorationRate;
    
    std::random_device rd;
    std::mt19937 gen;
    
public:
    RLAgent(const std::vector<State>& states, const std::vector<double>& actions,
            double lr = 0.1, double df = 0.9, double er = 0.1)
        : stateSpace(states), actionSpace(actions),
          learningRate(lr), discountFactor(df), explorationRate(er), gen(rd()) {
        
        for (const auto& state : stateSpace) {
            std::map<double, double> actionValues;
            for (double action : actionSpace) {
                actionValues[action] = 0.0;
            }
            qTable[state] = actionValues;
        }
    }
    
    double getAction(const State& state) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        if (dis(gen) < explorationRate) {
            std::uniform_int_distribution<> actionDis(0, actionSpace.size() - 1);
            return actionSpace[actionDis(gen)];
        }
        else {
            const auto& actions = qTable[state];
            double bestAction = 0.0;
            double maxValue = -std::numeric_limits<double>::infinity();
            
            for (const auto& [action, value] : actions) {
                if (value > maxValue) {
                    maxValue = value;
                    bestAction = action;
                }
            }
            return bestAction;
        }
    }
    
    void updateQTable(const State& currentState, double action, 
                     double reward, const State& nextState) {
        double bestNextValue = -std::numeric_limits<double>::infinity();
        for (const auto& [nextAction, value] : qTable[nextState]) {
            if (value > bestNextValue) {
                bestNextValue = value;
            }
        }
        
        double currentQ = qTable[currentState][action];
        double newQ = currentQ + learningRate * (reward + discountFactor * bestNextValue - currentQ);
        qTable[currentState][action] = newQ;
    }

    auto train(const int numEpisodes, const int stepsPerEpisode) {
        for (int episode = 0; episode < numEpisodes; ++episode) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> stateDis(0, stateSpace.size() - 1);
            State currentState = stateSpace[stateDis(gen)];
            
            for (int step = 0; step < stepsPerEpisode; ++step) {
                double action = getAction(currentState);
                double reward = -currentState.second; 
                State nextState = stateSpace[stateDis(gen)];  
                updateQTable(currentState, action, reward, nextState);
                currentState = nextState;
            }
        }
        return qTable;
    }
    
    void printQTable() const {
        for (const auto& [state, actions] : qTable) {
            std::cout << "State: (T=" << state.first << ", DR=" << state.second << ")\n";
            for (const auto& [action, value] : actions) {
                std::cout << "  Action " << std::setw(4) << action << ": Q value = " 
                          << std::fixed << std::setprecision(4) << value << "\n";
            }
            std::cout << "\n";
        }
    }
};

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
    std::shared_ptr<Config> conf_;
    std::thread monitor_update_thread_;
    std::thread rl_enhance_thread_;
    std::mutex mtx_;
    double update_time_gap;
    bool isfinish = false;
    int k_;

    std::shared_ptr<Filter> filter_;
    uint32_t update_filter_thres;
    bool update_flag;
    std::shared_ptr<std::mutex> mtx_1_;
    std::shared_ptr<std::condition_variable> cv_1_;
    bool* is_notify_;

    std::vector<ZoneState> zones_state_{};
    std::vector<std::string> ava_invul_backend_ips_{};
    std::vector<std::string> ava_vul_backend_ips_{};
    std::vector<std::string> backend_ips_{};
    std::unordered_set<std::string> new_unava_backend_ips_{};
    std::unordered_map<std::string, std::uint32_t> ip_2_zone{};

    std::unordered_map<uint32_t, std::vector<std::uint32_t>> region_to_zones_{};
    std::unordered_map<uint32_t, uint32_t> zone_to_region_{};

    uint32_t history_length_;
    std::unordered_map<uint32_t, std::deque<int>> zone_history_{};
    uint32_t top_k_;

    std::queue<uint32_t> volatile_set_{};
    uint32_t volatile_maximum_num_;
    bool cee_;

    double precision_;
    double recall_;
    double last_precision_;
    double last_recall_;
    double truly_preempt_;
    double predict_preempt_;
    double true_positive_;
    double eta_1_;
    double eta_2_;

    std::string flag_algorithm_;

    uint32_t arima_max_p;
    uint32_t arima_max_d;
    uint32_t arima_max_q;

    std::unordered_map<std::string, std::unordered_map<uint32_t, bool>> querys_state{}; //[backendip, [query_id, is_not_broken]]
    std::unordered_map<uint32_t, uint32_t> querys_to_stripes{};//[query_id, stripe_id]
    std::unordered_set<uint32_t> _cdc_querys{};
    std::unordered_map<uint32_t, bool> stripes_state{}; //[stripe_id, is_not_broken]
    std::mutex mtx_querys_state_;
    std::mutex mtx_stripes_state_;

    std::map<State, std::map<double, double>> qTable_;

    // std::vector<std::vector<uint32_t>> zone_preempt_hist{}; // the preemption history info of each zone 

    std::vector<int> zone_preempt_hist{}; 
    std::vector<std::vector<uint64_t>> C_;

    auto generateCombinations(int maxN) -> std::vector<std::vector<uint64_t>> {
        std::vector<std::vector<uint64_t>> C(maxN + 1, std::vector<uint64_t>(maxN + 1, 0));
        
        for (int n = 0; n <= maxN; ++n) {
            C[n][0] = C[n][n] = 1;
            for (int k = 1; k < n; ++k) {
                C[n][k] = C[n-1][k-1] + C[n-1][k];
            }
        }
        return C;
    }

    auto computeRedundancy(double ratio) -> double {
        return (1-ratio) + ratio * static_cast<double>(1 / static_cast<double>(conf_->k));
    }

    auto getRatio(int n, int k, int f) -> double {
        double C_1 = static_cast<double>(C_[n][f]);
        double C_2 = static_cast<double>(C_[n-k][f]);
        double divisor = (1-C_2/C_1-1/k+1);
        return 1 / divisor;
    }

    template<typename T>
    T findMax(const std::vector<T>& vec) {
        if (vec.empty()) {
            throw std::invalid_argument("Vector is empty");
        }
        
        auto maxIt = std::max_element(vec.begin(), vec.end());
        return *maxIt;
    }

    void updateZoneHistory(const uint32_t zone, int preemptCount) {
        auto& history = zone_history_[zone];
        if (history.size() == history_length_)
            history.pop_front();  // pop the least recent data
        history.push_back(preemptCount);  // push the new data
    }

    
    double pearsonCorrelation(const std::deque<int>& x, const std::deque<int>& y) {
        if (x.size() != y.size() || x.empty()) return 0.0;

        int n = x.size();
        double sum_x = 0, sum_y = 0;
        double sum_x2 = 0, sum_y2 = 0;
        double sum_xy = 0;

        for (int i = 0; i < n; ++i) {
            sum_x += x[i];
            sum_y += y[i];
            sum_x2 += x[i] * x[i];
            sum_y2 += y[i] * y[i];
            sum_xy += x[i] * y[i];
        }

        double numerator = n * sum_xy - sum_x * sum_y;
        double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
        if (denominator == 0) return 0.0;

        return numerator / denominator;
    }

    auto computeCorrelationMatrix(const std::unordered_map<uint32_t, std::deque<int>>& historyMap) ->
        std::map<std::pair<uint32_t, uint32_t>, double> {
        std::map<std::pair<uint32_t, uint32_t>, double> result;
        for (auto it1 = historyMap.begin(); it1 != historyMap.end(); ++it1) {
            for (auto it2 = std::next(it1); it2 != historyMap.end(); ++it2) {
                double corr = pearsonCorrelation(it1->second, it2->second);
                result[{it1->first, it2->first}] = corr;
            }
        }
        return result;
    }

    /*
     * brief: obtain the top-k most correlated zones of zone id
     */
    std::vector<uint32_t> top_k_correlated_zones(const std::unordered_map<uint32_t, std::deque<int>>& zone_history, const uint32_t& zone_id, int k) {
        std::vector<std::pair<uint32_t, double>> correlations;

        const auto& target_history = zone_history.at(zone_id);

        for (const auto& [other_zone, history] : zone_history) {
            if (other_zone == zone_id) continue;
            double corr = pearsonCorrelation(target_history, history);
            correlations.emplace_back(other_zone, corr);
        }

        sort(correlations.begin(), correlations.end(),
            [](const std::pair<uint32_t, double>& a, const std::pair<uint32_t, double>& b) {
                return a.second > b.second;
            });

        std::vector<uint32_t> top_k;
        for (int i = 0; i < std::min(k, (int)correlations.size()); ++i) {
            top_k.push_back(correlations[i].first);
        }

        return top_k;
    }

    void rl_based_tuning() {
        std::vector<State> states;
        std::vector<double> Ts = {};
        std::vector<double> DRs = {};

        for (double i = 0; i < 12; i += 0.5) {
            Ts.push_back(i);
        }

        for (double i = 0; i < 1; i += 0.05) {
            DRs.push_back(i);
        }
        
        for (double t : Ts) {
            for (double dr : DRs) {
                states.emplace_back(t, dr);
            }
        }
        
        std::vector<double> actions = {-1, 0, 1};
        
        RLAgent agent(states, actions);

        qTable_ = agent.train(1000, 10);
    }

public:
    Monitor() = delete;
    Monitor(std::shared_ptr<Config> conf,
            std::shared_ptr<Filter> filter,
            std::shared_ptr<std::mutex> mtx_1,
            std::shared_ptr<std::condition_variable> cv_1,
            bool* is_notify):
            conf_(conf),
            filter_(filter),
            mtx_1_(mtx_1),
            cv_1_(cv_1),
            is_notify_(is_notify) { 

        int zone_id = 0;
        history_length_ = conf_->history_length;
        top_k_ = conf_->top_k;
        precision_ = 0;
        recall_ = 0;
        last_precision_ = 0;
        last_recall_ = 0;
        for(auto region_trace: conf_->backend_IPs) {
            auto [trace_file_path, start_time_slice, region_id, ip_list] = region_trace;
            if (region_to_zones_.find(region_id) == region_to_zones_.end()) 
                region_to_zones_[region_id] = std::vector<std::uint32_t>();
            region_to_zones_[region_id].push_back(zone_id);
            zone_to_region_[zone_id] = region_id;
            auto trace_data = readJsonFile(trace_file_path).get("data", Json::Value());
            if(trace_data.isArray()) {
                traces_info_.emplace_back(trace_data, start_time_slice, ip_list);
            }
            backend_ips_.insert(backend_ips_.end(), ip_list.begin(), ip_list.end());
            for (auto backend_ip: ip_list) {
                ip_2_zone[backend_ip] = zone_id;
            }
            // initialize the zones and nodes
            zones_state_.emplace_back(zone_id, region_id, ip_list.size(), conf_->recovery_time, conf_->to_vul_num);

            zone_id++;
            std::cout << "zoneid: " << zone_id << std::endl;
        }
        flag_algorithm_ = conf_ -> flag_algorithm;
        update_time_gap = conf_ -> update_time_gap;

        arima_max_p = conf_->arima_max_p;
        arima_max_d = conf_->arima_max_d;
        arima_max_q = conf_->arima_max_q;

        update_filter_thres = 0;
        update_flag = false;
        k_ = conf_->k;
        querys_state.reserve(backend_ips_.size());

        // zone_preempt_hist.reserve(traces_info_.size());
        for (int i = 0; i < traces_info_.size(); i++) {
            zone_preempt_hist.emplace_back(0);
        }

        for (int i = 0; i < traces_info_.size(); i++) {
            zone_history_[i] = std::deque<int>(history_length_, 0);
        }

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

        C_ = generateCombinations(conf_->node_number);

        if (flag_algorithm_ == "fgd") {
            volatile_maximum_num_ = zone_id / (k_+1);
        }

        std::cout << "volatile_maximum_num_: " << volatile_maximum_num_ << std::endl;

        cee_ = conf_->cee;

        if (conf_->rl_enchance) {
            rl_enhance_thread_ = std::thread(&Monitor::rl_based_tuning, this);
        }

        LOG_INFO("Monitor created.");
    }

    // Monitor(std::shared_ptr<Config> conf_,
    //         uint32_t start_time_slice,
    //         std::shared_ptr<std::mutex> mtx_1,
    //         std::shared_ptr<std::condition_variable> cv_1,
    //         bool* is_notify):
    //         mtx_1_(mtx_1),
    //         cv_1_(cv_1),
    //         is_notify_(is_notify) { 

    //     for(auto region_trace: conf_ -> backend_IPs) {
    //         auto [trace_file_path, start_time_slice, ip_list] = region_trace;
    //         auto trace_data = readJsonFile(trace_file_path).get("data", Json::Value());
    //         if(trace_data.isArray()) {
    //             traces_info_.emplace_back(trace_data, start_time_slice, ip_list);
    //         }
    //         // initialize the zones and nodes
    //         zones_state_.emplace_back(ip_list.size(), conf_->recovery_time, conf_->to_vul_num);
    //     }
    //     flag_algorithm_ = conf_ -> flag_algorithm;

    //     arima_max_p = conf_->arima_max_p;
    //     arima_max_d = conf_->arima_max_d;
    //     arima_max_q = conf_->arima_max_q;

    //     findStartTime();
    //     SelectAvaBackendIPs();
    //     monitor_thread_ = std::thread(&Monitor::Update, this);
    //     LOG_INFO("Monitor created.");
    // }

    ~Monitor() {
        monitor_update_thread_.join();
        if (conf_->rl_enchance) {
            rl_enhance_thread_.join();
        }
    }

    // void findStartTime() {
    //     int min_size = 0x10000000;
    //     for(int i = 0; i < traces_info_.size(); i++) {
    //         min_size = std::min(static_cast<int>(traces_info_[i].trace_data.size()), min_size);
    //     }
    //     for(; cur_time_slice_ < min_size; cur_time_slice_++) {
    //         bool is_all_available = true;
    //         for(int j = 0; j < traces_info_.size(); j++) {
                
    //             if(traces_info_[j].trace_data[cur_time_slice_].asUInt() != traces_info_[j].ip_list.size()) {
    //                 is_all_available = false;
    //                 break;
    //             }
                    
    //         }
    //         if(is_all_available) break;
    //     }
    //     assert(cur_time_slice_ < min_size);

    //     LOG_INFO("Start time slice: %d", cur_time_slice_);
    // }

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
            if(isfinish) break;
            // std::cout << 1 << std::endl;

            if(flag_algorithm_ != "baseline")
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

     /*
     * @brief judge whether current time slice of region i is in the frequent period or not
     * @note use algorithm to find the frequent period at the beginning or use machine learning alg.
     */
    bool isInFreqPeriod(uint32_t cur_time_slice_, uint32_t region_id) {
        return false;
    }

    void updateZonesState() {
        for(int i = 0; i < zones_state_.size(); i++) {
            auto& info = traces_info_[i];
            info.cur_time_slice_++;
            LOG_INFO("Update zone %d, current time slice:%d", i, info.cur_time_slice_);
            if(info.cur_time_slice_ >= info.trace_data.size()) {
                isfinish = true;
                break;
            }
            auto cur_node_num = info.trace_data[info.cur_time_slice_].asUInt();
            auto last_node_num = info.trace_data[info.cur_time_slice_-1].asUInt();
            // std::cout << "0" << std::endl;
            zones_state_[i].updateZoneEachTime(cur_node_num);

            // std::cout << "1" << std::endl;
            zone_preempt_hist[i] = last_node_num - cur_node_num;
            updateZoneHistory(i, last_node_num - cur_node_num);
            // std::cout << "zone_preempt_hist[i]: " << zone_preempt_hist[i] << std::endl;
            // std::cout << "2" << std::endl;
        }
        if (conf_->filter_type == "auto") {
            uint32_t max_f = 0;

            try {
                // std::cout << "zone_preempt_hist:" << std::endl;
                // std::cout << "zone_preempt_hist.size:" << zone_preempt_hist.size() << std::endl;
                // for (int i = 0; i < zone_preempt_hist.size(); i++) {
                //     std::cout << zone_preempt_hist[i] << " ";
                // }
                // std::cout << std::endl;
                // max_f = findMax(zone_preempt_hist);
                max_f = std::accumulate(zone_preempt_hist.begin(), zone_preempt_hist.end(), 0);
                // std::cout << "max_f: " << max_f << std::endl;
            } catch (const std::invalid_argument& e) {
                max_f = 0;
            }

            max_f = max_f > 0 ? max_f : 0;
            double ratio = getRatio(conf_->node_number, conf_->k, max_f);

            // LOG_INFO("max_f is: %d", max_f);
            // LOG_INFO("CDC ratio is: %lf", ratio);
            
            filter_->setRatio(ratio);
        }
        
        // LOG_INFO("Update zones state completed!");
    }

    /*
     * @brief select backend ips of all regions, and devide them into invulnerable and vulnerable
     * @note update ava_invul_backend_ips_ and ava_vul_backend_ips_
     * @note std::move do not need copy
     */
    void SelectAvaBackendIPs() {
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
        // LOG_INFO("Select available backend ips completed!");
        return;
    }

    auto SelectAvaBackendIPsARegion(uint32_t region_id) const -> std::tuple<std::vector<std::string>,
                                                                 std::vector<std::string>, std::unordered_set<std::string>> {
        assert(region_id >= 0 && region_id < traces_info_.size());
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
        std::cout << "-------------------- region id: " << region_id << "-----------------------" << std::endl;
        std::cout << "ava nodes num:" << ava_nodes_id.size() << std::endl;
        std::cout << "available invulnerable size:" << ava_invul_ips.size() << std::endl;
        std::cout << "available vulnerable size:" << ava_vul_ips.size() << std::endl;

        std::cout << "ava nodes num:" << ava_nodes_id.size() << std::endl;
        std::cout << std::endl;
        std::cout << "available invulnerable size:" << ava_invul_ips.size() << std::endl;
        for(const auto& ip:ava_invul_ips) {
            std::cout << ip << std::endl;
        }
        std::cout << std::endl; 
        std::cout << "available vulnerable size:" << ava_vul_ips.size() << std::endl;
        for(const auto& ip:ava_vul_ips) {
            std::cout << ip << std::endl;
        }
        std::cout << std::endl;
        std::cout << "new preempted nodes size:" << new_unava_ips.size() << std::endl;
        for(const auto& ip:new_unava_ips) {
            std::cout << ip << std::endl;
        }
        std::cout << "-------------------------------------------------------------------------" << std::endl;

        return std::make_tuple(ava_invul_ips, ava_vul_ips, new_unava_ips);
    }

    void FlagZonePassive(uint32_t zone_id) {
        zones_state_[zone_id].updateFlagPassive();
    }

    void FlagZoneAll(uint32_t zone_id) {
        zones_state_[zone_id].updateFlagAll();
    }

    // auto FlagZoneRegionLevel(uint32_t zone_id) -> void {
    //     auto region_id = zone_to_region_[zone_id];
    //     for (auto zone_id: region_to_zones_[region_id]) {
    //         zones_state_[zone_id].updateFlagPassive();
    //     }
    // }

    // auto FlagZoneFineGrained(uint32_t zone_id) -> void {

    // }

    void FlagZoneActive(uint32_t zone_id) {
        auto pred_available_nodes_num = get_pred_ava_nodes_num(zone_id);
        LOG_INFO("Zone %d, Next Prediction: %d", zone_id, pred_available_nodes_num);
        zones_state_[zone_id].updateFlagPred(pred_available_nodes_num);
    }

    void EvictZoneFromVolatile(uint32_t zone_id) {
        zones_state_[zone_id].clearVolatileStatus();
    }

    void FlagZones() {
        std::unordered_map<uint32_t, bool> visit;
        // std::vector<int> visit_zones;
        std::unordered_map<uint32_t, int> visit_zones;
        std::unordered_map<uint32_t, int> true_zones;
        for(int i = 0; i < zones_state_.size(); i++) {
            auto info = traces_info_[i];
            auto cur_time_slice = info.cur_time_slice_;
            auto start_time_slice = info.start_time_slice_;

            // if(flag_algorithm_ == "auto") 
            //     flag_algorithm_ = isInFreqPeriod(cur_time_slice, i) ? "active" : "passive";

            if(flag_algorithm_ == "passive") {
                FlagZonePassive(i);
            }
            else if (flag_algorithm_ == "ldd") {
                auto region_id = zones_state_[i].get_region_id();
                std::cout << "------------ ldd -------------" << std::endl;
                std::cout << "region id: " << region_id << std::endl;
                // if (visit.find(region_id) == visit.end()) {
                bool if_preempt = zones_state_[i].ifPreempt();
                if (if_preempt) {
                    auto zones_in_region = region_to_zones_[region_id];
                    for (auto zone_id: zones_in_region) {
                        std::cout << "zone id: " << zone_id << std::endl;
                        FlagZoneAll(zone_id);
                    }
                }
                visit[region_id] = true;
                // }
                std::cout << "-------------------------------" << std::endl;
            }
            else if (flag_algorithm_ == "fgd") {
                bool if_preempt = zones_state_[i].ifPreempt();
                if (if_preempt) {
                    FlagZoneAll(i);
                    if (cee_) {
                        if (volatile_set_.size() == volatile_maximum_num_) {
                            int expired_zone = volatile_set_.back();
                            EvictZoneFromVolatile(expired_zone);
                            volatile_set_.pop();
                        }
                        volatile_set_.push(i);
                    }
                    // auto correlation_matrix = computeCorrelationMatrix(zone_history_);
                    // top-k zones
                    auto top_k_zones = top_k_correlated_zones(zone_history_, i, top_k_);
                    if (cee_) {
                        if (volatile_set_.size() == volatile_maximum_num_) {
                            for (int j = 0; j < top_k_; j++) {
                                int expired_zone = volatile_set_.back();
                                EvictZoneFromVolatile(expired_zone);
                                volatile_set_.pop();
                            }
                        }
                    }
                    for (auto zone: top_k_zones) {
                        FlagZoneAll(zone);
                        if (cee_) {
                            volatile_set_.push(zone);
                        }
                    }
                }
            }
            // else if (flag_algorithm_ == "cee") {
            //     bool if_preempt = zones_state_[i].ifPreempt();
            //     if (if_preempt) {
            //         true_zones[i] = zones_state_[i].preemptNum();
            //         if (visit_zones.find(i) == visit_zones.end()) {
            //             FlagZoneAll(i);
            //             visit_zones[i] = zones_state_[i].get_total_nodes_num();
            //             auto top_k_zones = top_k_correlated_zones(zone_history_, i, top_k_);
            //             for (auto zone: top_k_zones) {
            //                 FlagZoneAll(zone);
            //                 visit_zones[zone] = zones_state_[zone].get_total_nodes_num();
            //             }
            //         }
            //     }
            // }
            // else if(flag_algorithm_ == "active" && cur_time_slice - start_time_slice >= arima_max_p*2) // need to rethink!!!!
            //     FlagZoneActive(i);
            else {
                LOG_ERROR("Flag algorithm %s error", flag_algorithm_.c_str());
                exit(-1);
            }
            // LOG_INFO("Flag zone %d, current time slice:%d", i, cur_time_slice);
        }
        // update precision and recall
        if (flag_algorithm_ == "atk") {
            last_precision_ = precision_;
            last_recall_ = recall_;

            // precision
            double true_positive = 0;
            double predict_num = 0;
            for (auto it = visit_zones.begin(); it != visit_zones.end(); it++) {
                predict_num += it->second;
                if (true_zones.find(it->first) != true_zones.end()) {
                    true_positive += true_zones[it->first];
                }
            }
            precision_ = true_positive / predict_num;
        
            // recall
            double truly_preempt = 0;
            for (auto it = true_zones.begin(); it != true_zones.end(); it++) {
                truly_preempt += it->second;
            }
            recall_ = true_positive / truly_preempt;
            
            top_k_ = top_k_ - std::round(eta_1_*(recall_-last_recall_)) + std::round(eta_2_*(precision_-last_precision_));
        
            top_k_ = top_k_ < 0 ? 0 : top_k_;
            top_k_ = top_k_ > zones_state_.size() / 2 ? zones_state_.size() / 2 : top_k_;
        }
        
    }

    void FlagZoneInitialization() {
        for(int i = 0; i < zones_state_.size(); i++) {
            zones_state_[i].FlagInitialization();
        }
        LOG_INFO("Flag Zone Initialization completed!");
        
    }

    void FlagZoneInitialization(int zone_id) {
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

        // std::cout << "Next prediction: " << forecast[0] << std::endl;

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

    // auto get_zones_preempt_hist() -> std::vector<std::vector<uint32_t>> {return zone_preempt_hist;}

    // auto get_a_zone_preempt_hist(int region_id) -> std::vector<uint32_t> {return zone_preempt_hist[region_id];}

    // auto get_latest_a_zone_preempt_hist(int region_id, int duration) -> std::vector<uint32_t> {
    //     return std::vector<uint32_t>(zone_preempt_hist[region_id].end() - duration, zone_preempt_hist[region_id].end());
    // }

    auto get_last_preempt() -> std::vector<int> {
        // std::vector<uint32_t> res;
        // for (auto zone_preempt: zone_preempt_hist) {
        //     res.push_back(zone_preempt);
        // }
        // return res;
        return zone_preempt_hist;
    }

    bool findAvaBackendIPs() {
        std::lock_guard<std::mutex> lock(mtx_);
        if(ava_invul_backend_ips_.size() + ava_vul_backend_ips_.size() <= k_) return false;
        return true;
    }


    auto sendQueryToABackend(std::string backendip, uint32_t query_id, uint32_t encode_id, bool is_cdc_stripe) -> void {
        // std::cout << "send query "<< query_id << " to a backend " << backendip << " start!";
        {
            std::lock_guard<std::mutex> lock(mtx_querys_state_);
            querys_state[backendip][query_id] = true;
            if(is_cdc_stripe) {
                querys_to_stripes[query_id] = encode_id;
                _cdc_querys.insert(query_id);
            }
        }

        {
            std::lock_guard<std::mutex> lock(mtx_stripes_state_);
            if(is_cdc_stripe && stripes_state.find(encode_id) == stripes_state.end()) {
                stripes_state[encode_id] = true;

            }
        }
        // std::cout << "send query "<< query_id << " to a backend " << backendip << " completed!";
    }

    /*
     *@brief update querys state and stripes state
     */
    void UpdateQuerysState() {
        std::lock_guard<std::mutex> lock(mtx_querys_state_);
        for(const auto& ip: new_unava_backend_ips_) {
            // std::cout << "new unava ip:" << ip << std::endl;
            for(auto& query_state: querys_state[ip]) {
                if (query_state.second) {
                    std::cout << "set query id: " << query_state.first << " as false" << std::endl;
                }
                query_state.second = false;

                if(_cdc_querys.find(query_state.first) == _cdc_querys.end())
                    continue;
                    
                std::lock_guard<std::mutex> lock(mtx_stripes_state_);
                auto stripe_id = querys_to_stripes[query_state.first];
                if(stripes_state[stripe_id]) stripes_state[stripe_id] = false;
            }
        }

    }

    bool IsQueryBroken(std::string backendip, uint32_t query_id) {
        std::lock_guard<std::mutex> lock(mtx_querys_state_);
        return !querys_state[backendip][query_id];
    }

    bool IsStripeBroken(uint32_t stripe_id) {
        std::lock_guard<std::mutex> lock(mtx_stripes_state_);
        return !stripes_state[stripe_id];
    }

    void DeleteAQueryState(std::string backendip, uint32_t query_id) {
        std::lock_guard<std::mutex> lock(mtx_querys_state_);
        querys_state[backendip].erase(query_id);
        querys_to_stripes.erase(query_id);
    }

    void DeleteAStripeState(uint32_t stripe_id) {
        std::lock_guard<std::mutex> lock(mtx_stripes_state_);
        stripes_state.erase(stripe_id);
    }

    auto Ip2Region(std::string backend_ip) -> std::uint32_t {
        return ip_2_zone[backend_ip];
    }

    auto avaRegionNum() -> std::uint32_t {
        std::unordered_set<std::uint32_t> ava_region;
        for (auto ip: ava_invul_backend_ips_) {
            std::uint32_t region = ip_2_zone[ip];
            ava_region.insert(region);
        }
        for (auto ip: ava_vul_backend_ips_) {
            std::uint32_t region = ip_2_zone[ip];
            ava_region.insert(region);
        }
        return ava_region.size();
    }
};