#pragma once

#include"../inc/inc.hh"

struct NodeState {
    NodeFlag state_ = NodeFlag::INVULNERABLE;
    uint32_t time_slice_ = 0;   // The time elapsed from being invulnerable to being vulnerable
    uint16_t id_;   // node id indexed from "0"

    NodeState(uint16_t id): id_(id) {};
};

class ZoneState {

public:
    ZoneState(uint32_t total_nodes_num, uint32_t recovery_time, uint32_t to_vul_num){
        total_nodes_num_ = total_nodes_num;
        recovery_time_ = recovery_time;
        to_vul_num_ = to_vul_num;
        
        for(int i = 0; i < total_nodes_num; i++) {
            nodes_.emplace_back(i);
            available_nodes_id.push_back(i);
            invulnerable_nodes_id.insert(i);
        }
    }

    /*
    * @brief: update the vulnerable nodes' time, and decide to change it to invulnerable node or not.
    */
    void updateNodeTime(){
        for (auto it = vulnerable_nodes_id.begin(); it != vulnerable_nodes_id.end(); ) {
            auto index = *it;
            auto& node = nodes_[index];
            assert(node.state_ == NodeFlag::VULNERABLE);
            node.time_slice_++;
            if (node.time_slice_ == recovery_time_) {
                node.state_ = NodeFlag::INVULNERABLE;
                node.time_slice_ = 0;
                invulnerable_nodes_id.insert(node.id_);
                it = vulnerable_nodes_id.erase(it);
            } else {
                ++it;
            }
        } 
    }

    /*
    * @brief：update zone state each time slice, to a certain extent, it is the passive way.
    * @note: 1. update the time of vulterable nodes and decide to change it to invulterable or not
    *        2. if available nodes num is increasing, choose the unavailable nodes FIFO to be available
    *        3. otherwise, if available nodes num is decreasing, choose the available nodes FIFO to be unavailable,
    *           record the new_unavailable nodes_id
    *        
    */
   void updateZoneEachTime(int cur_available_nodes_num) {

        updateNodeTime();
        new_unavailable_nodes_id.clear();

        // std::cout << "cur available nodes num:" << cur_available_nodes_num << " ,available nodes num:" << available_nodes_id.size();
        if(cur_available_nodes_num >= available_nodes_id.size()) {
            auto num_to_move = cur_available_nodes_num - available_nodes_id.size();
            for (size_t i = 0; i < num_to_move; ++i) {
                auto node_id = unavailable_nodes_id.front();
                unavailable_nodes_id.pop_front();
                available_nodes_id.push_back(node_id);
            } 
        }

        else {
            auto num_to_move = available_nodes_id.size() - cur_available_nodes_num;
            for (size_t i = 0; i < num_to_move; ++i) {
                auto node_id = available_nodes_id.front();
                available_nodes_id.pop_front();
                unavailable_nodes_id.push_back(node_id);
                new_unavailable_nodes_id.insert(node_id);
            } 
        }
        assert(available_nodes_id.size() + unavailable_nodes_id.size() == total_nodes_num_);
   }

   /*
   * @brief: update the zones state passively
   * @note: judge whether the new_unavailable_node is invulnerable node,
   *        if so, choose to_vulnerable_num nodes
   */
  void updateFlagPassive() {
    bool is_need_flag = false;
    for(const auto& index: new_unavailable_nodes_id) {
        is_need_flag = true;
        if(invulnerable_nodes_id.find(index) != invulnerable_nodes_id.end()) {
            // is_need_flag = true;
            nodes_[index].state_ = NodeFlag::VULNERABLE;
            vulnerable_nodes_id.insert(index);
            invulnerable_nodes_id.erase(index);
            // break;
        }
    }

    if(!is_need_flag) return;

    auto vul_num = 0;
    for(const auto& index: available_nodes_id) {
        if(vul_num == to_vul_num_) break;
        if(invulnerable_nodes_id.find(index) != invulnerable_nodes_id.end()){
            nodes_[index].state_ = NodeFlag::VULNERABLE;
            vulnerable_nodes_id.insert(index);
            invulnerable_nodes_id.erase(index);
            vul_num++;
        }
    }
    assert(vulnerable_nodes_id.size() + invulnerable_nodes_id.size() == total_nodes_num_);
  }

   /*
   * @brief: update the zones state with ARIMA alg
   * @note: 1. if available nodes num is decreasing, choose the available nodes randomly to be unavailable,
   *           but actually not change its availability
   *        2. judge whether the above-mentioned node is invulnerable node,
   *           if so, choose to_vulnerable_num nodes, change its flag
   */
   void updateFlagPred(int pred_available_nodes_num) {

    bool is_need_flag = false;
    if(pred_available_nodes_num < available_nodes_id.size()) {
        // auto pred_unava_nodes_num = available_nodes_id.size() - pred_available_nodes_num;
        is_need_flag = true;
        // for (size_t i = 0; i < pred_unava_nodes_num; ++i) {
        //     // pred_unavailable_nodes_id.insert(available_nodes_vec[i]);
        //     auto index = available_nodes_id.at(i);
        //     if(nodes_[index].state_ == NodeFlag::INVULNERABLE) {
        //         // is_need_flag = true;
        //         break;
        //     }
        // } 
    }

    if(!is_need_flag) return;

    auto vul_num = 0;
    for(const auto& index: available_nodes_id) {
        if(vul_num == to_vul_num_) break;
        if(invulnerable_nodes_id.find(index) != invulnerable_nodes_id.end()){
            nodes_[index].state_ = NodeFlag::VULNERABLE;
            vulnerable_nodes_id.insert(index);
            invulnerable_nodes_id.erase(index);
            vul_num++;
        }
    }
    assert(vulnerable_nodes_id.size() + invulnerable_nodes_id.size() == total_nodes_num_);

   }

   auto FlagInitialization() -> void {
        auto start_node_id = available_nodes_id.front();
        assert(nodes_[start_node_id].state_ == NodeFlag::INVULNERABLE);
        nodes_[start_node_id].state_ = NodeFlag::VULNERABLE;
        invulnerable_nodes_id.erase(start_node_id);
        vulnerable_nodes_id.insert(start_node_id);
   }

   auto get_available_nodes_id() const -> std::unordered_set<uint32_t> {return std::unordered_set<uint32_t>(available_nodes_id.begin(), available_nodes_id.end());}
   auto get_unavailable_nodes_id() const -> std::unordered_set<uint32_t> {return std::unordered_set<uint32_t>(unavailable_nodes_id.begin(), unavailable_nodes_id.end());}
   auto get_vulnerable_nodes_id() const -> std::unordered_set<uint32_t> {return vulnerable_nodes_id;}
   auto get_invulnerable_nodes_id() const -> std::unordered_set<uint32_t> {return invulnerable_nodes_id;}
   auto get_new_unavailable_nodes_id() const -> std::unordered_set<uint32_t> {return new_unavailable_nodes_id;} 
   auto get_available_nodes_num() const -> uint32_t { return available_nodes_id.size();}

   /*
    *@brief: get the node id depond on the preemption probability for parity task
    */
   auto get_ava_node_id_decrease(double alpha_d) const -> uint32_t {
        int n = available_nodes_id.size()-1;
        if (n == 0) {
            throw std::runtime_error("The deque is empty!");
        }

        // calculate the sum of probability
        // int sum = n * (n + 1) / 2;
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += std::exp(-alpha_d * i);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double randomValue = dis(gen);

        double cumulativeProbability = 0.0;
        for (int i = 0; i < n; ++i) {
            // calculate cur elem probability
            // double currentProbability = static_cast<double>(n - i) / sum;
            double currentProbability = std::exp(-alpha_d * i) / sum;
            cumulativeProbability += currentProbability;
            if (randomValue < cumulativeProbability) {
                return available_nodes_id[i];
            }
        }
        return available_nodes_id.back();
   }

    /*
    *@brief: get the node id depond on the preemption probability for orignal task
    */
   auto get_ava_node_id_increase(double alpha_i) const -> uint32_t {
        int n = available_nodes_id.size();
        if (n == 0) {
            throw std::runtime_error("The deque is empty!");
        }

        // calculate the sum of probability
        // int sum = n * (n + 1) / 2;
        double sum = 0.0;
        for (int i = 1; i < n; ++i) {
            sum += std::pow(i, alpha_i);
        }

        if (sum == 0.0) {
            return available_nodes_id.front();
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double randomValue = dis(gen);

        double cumulativeProbability = 0.0;
        for (int i = 1; i < n; ++i) {
            // calculate cur elem probability
            // double currentProbability = static_cast<double> (i + 1) / sum;
            double currentProbability = std::pow(i, alpha_i) / sum;
            cumulativeProbability += currentProbability;
            if (randomValue < cumulativeProbability) {
                return available_nodes_id[i];
            }
        }
        return available_nodes_id.back();
    }

    auto get_node_id() const -> uint32_t {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, nodes_.size() - 1);
        return dis(gen);
    }

    auto get_ava_node_id() const -> uint32_t {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, available_nodes_id.size() - 1);
        return dis(gen);
    }

private:
    uint32_t total_nodes_num_ = 0;
    std::vector<NodeState> nodes_ = {};
    // std::unordered_set<uint32_t> available_nodes_id = {};
    // std::unordered_set<uint32_t> unavailable_nodes_id = {};
    std::deque<uint32_t> available_nodes_id = {};
    std::deque<uint32_t> unavailable_nodes_id = {};
    std::unordered_set<uint32_t> vulnerable_nodes_id = {};
    std::unordered_set<uint32_t> invulnerable_nodes_id = {};
    std::unordered_set<uint32_t> new_unavailable_nodes_id = {};
    uint32_t recovery_time_ = 0;
    uint32_t to_vul_num_ = 0;
};