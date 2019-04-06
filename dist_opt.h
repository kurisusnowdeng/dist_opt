#ifndef DIST_OPT_H
#define DIST_OPT_H

#include <iostream>
#include <deque>
// #include <tuple>
#include <utility>
#include <mutex>
#include <condition_variable>

typedef float DATA_TYPE;
typedef int COMM;
typedef DATA_TYPE* PTR;

// test optimizer
class Optimizer {
    public:
        Optimizer(float rate) {
            lr = rate;
        }

        void update(PTR keys[], DATA_TYPE grads[], int num) {
            for (int i = 0; i < num; i++) {
                *(keys[i]) -= lr * grads[i];
            }
        }

        float get_lr() {
            return lr;
        }
    private:
        float lr;
};

// test do_allreduce
// int do_allreduce(DATA_TYPE* sendbuff, DATA_TYPE* receivebuff, size_t size, COMM& comm);

// test allreduce_wait
// bool allreduce_wait();

class Dist_opt {
    public:
        // construtor
        Dist_opt(size_t thr, COMM &comm, Optimizer &opt);

        // destructor
        // ~Dist_opt();

        // push parameters into combiner queue
        void update(PTR keys[], DATA_TYPE grads[], int num);

        // move parameters from combiner queue to buff and call allreduce 
        void combine();

        // check if parameters are ready
        // bool wait(PTR keys[]);

        // initialize buff
        DATA_TYPE* init_buff(size_t thr);

        // get the total size of items in the queue
        size_t get_size();

        // check if combiner is full
        bool full(size_t s);

    private:
        std::deque<std::pair<PTR, DATA_TYPE>> combiner;
        DATA_TYPE* combiner_buff;
        size_t size;
        size_t threshold;
        std::mutex mtx;
        std::condition_variable pop_all;
        std::condition_variable push;
        Optimizer optimizer;
        int communicator;
};

#endif