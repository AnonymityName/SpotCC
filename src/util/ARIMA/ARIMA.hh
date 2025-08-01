#pragma once

#include <Eigen/Dense>
#include <numeric>
#include <fstream>
#include <limits>
#include <vector>

class ARIMA {
public:
    ARIMA(int p, int d, int q);
    void fit(const std::vector<double>& series);
    double predict(const std::vector<double>& series) const;
    std::vector<double> forecast(const std::vector<double>& series, int steps) const;
    double aic(const std::vector<double>& series) const;

private:
    int p_, d_, q_;
    Eigen::VectorXd ar_params_;
    double ma_param_;
    double last_error_;

    std::vector<double> difference(const std::vector<double>& data, int d) const;
    std::vector<double> invert_difference(const std::vector<double>& diff, double first) const;
    void build_training_data(const std::vector<double>& series, Eigen::MatrixXd& X, Eigen::VectorXd& y) const;
};

ARIMA select_best_model(const std::vector<double>& data, int max_p, int max_d, int max_q);
void save_forecast_csv(const std::vector<double>& original, const std::vector<double>& forecast);