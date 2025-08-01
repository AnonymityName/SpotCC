#include <iostream>
#include "ARIMA.hh"
#include <chrono>

int main() {
    std::vector<double> data = {
        112,118,132,129,121,135,148,148,136,119,104,118,
        115,126,141,135,125,149,170,170,158,133,114,140
    };

    auto start = std::chrono::high_resolution_clock::now();
    ARIMA model = select_best_model(data, 12, 2, 2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Training time: " << duration << " ms" << std::endl;

    double next = model.predict(data);
    std::cout << "Next prediction: " << next << std::endl;

    std::vector<double> forecast = model.forecast(data, 5);
    std::cout << "Next 5 forecasts: ";
    for (double val : forecast)
        std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}