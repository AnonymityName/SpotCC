#include "ARIMA.hh"

ARIMA::ARIMA(int p, int d, int q) : p_(p), d_(d), q_(q), ma_param_(0.0), last_error_(0.0) {}

std::vector<double> ARIMA::difference(const std::vector<double>& data, int d) const {
    std::vector<double> result = data;
    for (int i = 0; i < d; ++i) {
        std::vector<double> temp;
        for (size_t j = 1; j < result.size(); ++j)
            temp.push_back(result[j] - result[j - 1]);
        result = temp;
    }
    return result;
}

std::vector<double> ARIMA::invert_difference(const std::vector<double>& diff, double first) const {
    std::vector<double> result = {first};
    for (double d : diff)
        result.push_back(result.back() + d);
    return result;
}

void ARIMA::build_training_data(const std::vector<double>& series, Eigen::MatrixXd& X, Eigen::VectorXd& y) const {
    int n = series.size() - p_;
    X.resize(n, p_);
    y.resize(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p_; ++j)
            X(i, j) = series[i + j];
        y(i) = series[i + p_];
    }
}

void ARIMA::fit(const std::vector<double>& series) {
    std::vector<double> diffed = difference(series, d_);
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    build_training_data(diffed, X, y);

    ar_params_ = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    Eigen::VectorXd residuals = y - X * ar_params_;
    if (q_ > 0 && residuals.size() > 1) {
        ma_param_ = residuals.tail(residuals.size() - 1).dot(residuals.head(residuals.size() - 1)) /
                    residuals.head(residuals.size() - 1).squaredNorm();
    }
    last_error_ = residuals[residuals.size() - 1];
}

double ARIMA::predict(const std::vector<double>& series) const {
    std::vector<double> diffed = difference(series, d_);
    double ar_sum = 0.0;
    for (int i = 0; i < p_; ++i)
        ar_sum += ar_params_[i] * diffed[diffed.size() - p_ + i];
    double prediction = ar_sum + ma_param_ * last_error_;
    return series.back() + prediction;
}

std::vector<double> ARIMA::forecast(const std::vector<double>& series, int steps) const {
    std::vector<double> history = series;
    std::vector<double> forecasted;
    for (int i = 0; i < steps; ++i) {
        double next = predict(history);
        forecasted.push_back(next);
        history.push_back(next);
    }
    return forecasted;
}

double ARIMA::aic(const std::vector<double>& series) const {
    std::vector<double> diffed = difference(series, d_);
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    build_training_data(diffed, X, y);

    Eigen::VectorXd residuals = y - X * ar_params_;
    double rss = residuals.squaredNorm();
    int n = y.size();
    int k = p_ + q_;
    return n * std::log(rss / n) + 2 * k;
}

ARIMA select_best_model(const std::vector<double>& data, int max_p, int max_d, int max_q) {
    double best_aic = std::numeric_limits<double>::max();
    ARIMA best_model(0, 0, 0);

    for (int p = 0; p <= max_p; ++p) {
        for (int d = 0; d <= max_d; ++d) {
            for (int q = 0; q <= max_q; ++q) {
                try {
                    ARIMA model(p, d, q);
                    model.fit(data);
                    double aic = model.aic(data);
                    if (aic < best_aic) {
                        best_aic = aic;
                        best_model = model;
                    }
                } catch (...) {
                    continue;
                }
            }
        }
    }

    return best_model;
}

void save_forecast_csv(const std::vector<double>& original, const std::vector<double>& forecast) {
    std::ofstream file("forecast.csv");
    file << "original,forecast\n";
    size_t max_len = std::max(original.size(), original.size() + forecast.size());
    for (size_t i = 0; i < max_len; ++i) {
        if (i < original.size())
            file << original[i];
        else
            file << "";

        file << ",";
        if (i >= original.size())
            file << forecast[i - original.size()];
        file << "\n";
    }
    file.close();
}