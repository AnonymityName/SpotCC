#pragma once
#include <list>
#include <unordered_map>
#include <algorithm>
#include <mutex>
#include <iostream>
#include <cassert>
#include "../util/json/json.h"
#include "../common/logger.hh"
using namespace std;


template<typename K, typename V>
class BasicCache {
public:
    BasicCache(const Json::Value& config) { 
        use_cache_ = config.get("use_cache", false).asBool();
        if (use_cache_) {
            capacity_  = config.get("capacity", 0).asInt();
            assert(capacity_ > 0 && "capacity should be greater than 0");
        }
    }
    
    virtual bool get(const K& key, V& value) = 0;

    virtual void put(const K& key, const V& value) = 0;

    std::pair<double, double> getCacheInfo() {
        std::lock_guard<std::mutex> lck(mutex_);
        if(cache_info_.hit + cache_info_.miss == 0) {
            return {0, 0};
        }
        double hitRate = (double)cache_info_.hit / (cache_info_.hit + cache_info_.miss);
        double missRate = (double)cache_info_.miss / (cache_info_.hit + cache_info_.miss);
        return {hitRate, missRate};
    }

protected:
    bool use_cache_;
    int capacity_;

    unordered_map<K, typename list<V>::iterator> map_;
    list<V> list_;
    std::mutex mutex_;
    struct cache_info {
        int miss = 0;
        int hit = 0;
    } cache_info_;
};

template <typename K, typename V>
class LruCache : public BasicCache<K, V> {

public:
    LruCache(const Json::Value& config) : BasicCache<K, V>(config) {}

    bool get(const K& key, V& value) override {
        if (!use_cache_) {
            return false;
        }
        std::lock_guard<std::mutex> lck(this->mutex_);
        if(this->map_.find(key) == map_.end()) {
            this->cache_info_.miss++;
            return false;
        }
        list_.splice(list_.begin(), list_, map_[key]);
        value = *map_[key];
        cache_info_.hit++;
        return true;
    }

    void put(const K& key, const V& value) override {
        if (!use_cache_) {
            return ;
        }
        std::lock_guard<std::mutex> lck(mutex_);
        if(map_.find(key) != map_.end()) {
            *map_[key] = value;
            list_.splice(list_.begin(), list_, map_[key]);
        } else {
            if(map_.size() < capacity_) {
                map_[key] = list_.insert(list_.begin(), value);
            } else {
                map_.erase(find_if(map_.begin(), map_.end(), [&](const auto x) {
                    return x.second == (--list_.end());
                }));
                list_.pop_back();
                map_[key] = list_.insert(list_.begin(), value);
            }
        }
    }

    
    using BasicCache<K, V>::use_cache_;
    using BasicCache<K, V>::capacity_;
    using BasicCache<K, V>::map_;
    using BasicCache<K, V>::list_;
    using BasicCache<K, V>::mutex_;
    using BasicCache<K, V>::cache_info_;
};