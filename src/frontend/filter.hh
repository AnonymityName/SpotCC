#pragma once
#include "../inc/inc.hh"
// #include "monitor2parts.hh"

class Filter {
private:
    std::shared_ptr<Config> conf_;
    // std::shared_ptr<Monitor> monitor_;
    double backup_ratio_ = 0.5;
    double cdc_ratio_;
    std::shared_ptr<std::mutex> mtx_ = std::make_shared<std::mutex>(); 
    std::vector<std::vector<uint64_t>> C_;
    double ratio_;
    int update_interval_;
    bool update_enable_;
    double last_ratio_;

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

public:
    Filter(std::shared_ptr<Config> conf) {
        conf_ = conf;
        cdc_ratio_ = conf_->cdc_ratio;
        update_interval_ = 0;
        update_enable_ = true;
        last_ratio_ = 1;
        ratio_ = 1;

        if (conf_->flag_algorithm == "baseline") cdc_ratio_ = 80;

        C_ = generateCombinations(conf_->node_number);

    }
    /**
     * @brief choose the encoding strategy (Backup or CDC)
     *        use the random number 
    **/
    void filterWorker(EncodeType& encodeType) {
        std::random_device rd; 
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 100);
        if (conf_->filter_type == "manual") { // manual mode
            if(dis(gen) > cdc_ratio_) {
                encodeType = EncodeType::Backup;
                LOG_INFO("Choose the backup queue.");
            }
            else {
                encodeType = EncodeType::CDC;
                LOG_INFO("Choose the CDC queue.");
            }
        }
        else { // automatic mode
            // std::vector<uint32_t> f_list = monitor_->get_last_preempt();
            // uint32_t max_f = 0;

            // try {
            //     max_f = findMax(f_list);
            // } catch (const std::invalid_argument& e) {
            //     max_f = 0;
            // }

            // max_f = max_f > 0 ? max_f : 0;
            // double ratio = getRatio(conf_->node_number, conf_->k, max_f);

            // LOG_INFO("max_f is: %d", max_f);
            LOG_INFO("CDC ratio in filter is: %lf", ratio_);

            if(dis(gen) > ratio_ * 100) {
                encodeType = EncodeType::Backup;
                LOG_INFO("Choose the backup queue.");
            } else {
                encodeType = EncodeType::CDC;
                LOG_INFO("Choose the CDC queue.");
            }       
        }   
    } 

    void setRatio(double ratio) {
        bool last_same = ratio == ratio_;

        if (update_enable_ && !last_same) {
            ratio_ = ratio;
            update_interval_ = 0;
            update_enable_ = false;
        }
        else if (!update_enable_ && update_interval_ < 10) {
            update_interval_++;
        }
        else if (!update_enable_ && update_interval_ == 10) {
            update_interval_ = 0;
            update_enable_ = true;
        }
    }

    void updateFilterRatio(double CDC_recovery_time, double Rep_recovery_time) {
        auto backup_redundancy = conf_ -> backup_num;
        auto cdc_redundancy = 1.0 / conf_ -> k;

        // z-score
        auto res1 = stdTwoVar(cdc_redundancy, backup_redundancy);
        auto cdc_redundancy_ndh = res1.first;
        auto backup_redundancy_ndh = res1.second;

        auto res2 = stdTwoVar(CDC_recovery_time, Rep_recovery_time);
        auto cdc_recovery_time_ndh = res2.first;
        auto backup_recovery_time_ndh = res2.second;

        // cost == recovery
        {
            std::unique_lock<std::mutex> lock(*mtx_);
            backup_ratio_ = (cdc_recovery_time_ndh - cdc_redundancy_ndh) / 
                            (backup_redundancy_ndh - cdc_redundancy_ndh - 
                                backup_recovery_time_ndh + cdc_recovery_time_ndh);
        }
        
    }

    auto stdTwoVar(double x1, double x2) -> std::pair<double, double> {
        auto average = (x1 + x2) / 2.0;
        auto std_dev = std::sqrt(std::pow(x1 - average, 2) 
                                    + std::pow(x2 - average, 2));
        auto x1_ndh = (x1 - average) / std_dev;
        auto x2_ndh = (x2 - average) / std_dev;

        return std::make_pair(x1_ndh, x2_ndh);
    }

};
