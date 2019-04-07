#ifndef DIST_OPT_H
#define DIST_OPT_H

#include <iostream>
#include <vector>
#include <deque>
#include <unordered_map>
#include <string>
#include <utility>
#include <thread>
#include <mutex>
#include <condition_variable>

typedef float DATA_TYPE;
typedef int COMM;
typedef std::vector<std::pair<std::string, DATA_TYPE>> GRAD_LIST;
typedef std::unordered_map<std::string, DATA_TYPE> PARA_MAP;

// test optimizer
class Optimizer {
    public:
        Optimizer() {
            parameters = nullptr;
            lr = 0;
        }

        Optimizer(PARA_MAP &model, float rate) {
            parameters = &model;
            lr = rate;
        }

        Optimizer(const Optimizer &opt) {
            parameters = opt.parameters;
            lr = opt.lr;
        }

        void update(const GRAD_LIST &grads) {
            std::string key;
            DATA_TYPE val;
            for (auto item: grads) {
                std::tie(key, val) = item;
                (*parameters)[key] -= lr * val;
            }
        }

        float get_lr() {
            return lr;
        }

    private:
        PARA_MAP *parameters;
        float lr;
};

// test do_allreduce
// int do_allreduce(DATA_TYPE* sendbuff, DATA_TYPE* receivebuff, size_t size, COMM& comm);

// test allreduce_wait
// bool allreduce_wait();

// locks and signals
typedef struct {
    std::mutex mtx;
    std::condition_variable pop_all;
    std::condition_variable push;
} Context;


class Dist_opt {
    public:
        // construtor
        Dist_opt(size_t thr, const COMM &comm, const Optimizer &opt);

        // destructor
        ~Dist_opt();

        // push parameters into combiner queue
        void update(int id, const GRAD_LIST &grads);

        // move parameters from combiner queue to buff and call allreduce 
        void combine();

        // check if parameters are ready
        bool wait(const std::vector<std::string> &keys);

        bool wait(const std::string &key);

        // initialize buff
        DATA_TYPE* init_buff(size_t thr);

        // get the total size of items in the queue
        size_t get_size();

        // check if combiner is full
        bool full(size_t s);

    private:
        std::deque<std::pair<std::string, DATA_TYPE>> combiner;
        DATA_TYPE* combiner_buff;
        size_t size;
        size_t threshold;
        Optimizer optimizer;
        int communicator;
        Context *ctx;
        std::unordered_map<std::string, bool> dict;
        std::thread *combiner_thread;
};

#endif