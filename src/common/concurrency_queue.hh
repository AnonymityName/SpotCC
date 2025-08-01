#pragma once
#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

template <typename T>
class ConcurrencyQueue {
public:
  void Push(const T& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(value);
    lock.unlock();
    cv_.notify_one(); 
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); }); 

    T value = queue_.front();
    queue_.pop();
    return value;
  }

  T Front() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); }); 
    T value = queue_.front();
    
    return value;
  }

  T Back() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); }); 
    T value = queue_.back();
    
    return value;
  }

  size_t Size() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
  }

private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
};