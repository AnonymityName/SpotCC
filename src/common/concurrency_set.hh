#pragma once
#include <iostream>
#include <unordered_set>
#include <mutex>
#include <thread>

template <typename T>
class ConcurrencySet {
private:
    std::unordered_set<T> set;
    mutable std::mutex mtx;

public:
    void insert(const T& value) {
        std::lock_guard<std::mutex> lock(mtx);
        set.insert(value);
    }

    void erase(const T& value) {
        std::lock_guard<std::mutex> lock(mtx);
        set.erase(value);
    }

    bool contains(const T& value) const {
        std::lock_guard<std::mutex> lock(mtx);
        return set.find(value) != set.end();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return set.size();
    }
};