#ifndef DIST_OPT_H
#define DIST_OPT_H

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <utility>
#include <thread>
#include <mutex>
#include <condition_variable>

#define delete_ptr(p) { if ((p) != nullptr) { delete (p); p = nullptr; } }
#define delete_arr(p) { if ((p) != nullptr) { delete [] (p); p = nullptr; } }

typedef float DATA_TYPE;
typedef std::pair<DATA_TYPE*, size_t> PARA_TYPE;
typedef int COMM;
typedef std::vector<std::pair<std::string, PARA_TYPE>> GRAD_LIST;
typedef std::unordered_map<std::string, PARA_TYPE> PARA_MAP;

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
            DATA_TYPE *val;
            DATA_TYPE *new_val;
            size_t size;
            size_t new_size;
            PARA_TYPE temp;
            for (auto item: grads) {
                std::tie(key, temp) = item;
                std::tie(new_val, new_size) = temp;
                temp = (*parameters)[key];
                std::tie(val, size) = temp;
                printf("Start to update %s, size = %lu, new size = %lu\n", key.c_str(), size, new_size);
                printf("    %s updated to [", key.c_str());
                for (int i = 0; i * sizeof(DATA_TYPE) < size; i++) {
                    val[i] += lr * new_val[i];
                    printf("%.2f ", val[i]);
                }
                printf("]\n");
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

namespace dist_opt {

    // locks and signals
    extern std::mutex mtx;
    extern std::condition_variable pop_all;
    extern std::condition_variable push;

    class Dist_opt {
        public:
            // construtor
            Dist_opt(size_t threshold, size_t size, const COMM &comm, const Optimizer &opt);

            // destructor
            ~Dist_opt();

            // push parameters into combiner queue
            void update(int tID, const GRAD_LIST &grads);

            // move parameters from combiner queue to buff and call allreduce 
            void combine();

            // check if parameters are ready
            bool wait(const std::vector<std::string> &keys);

            bool wait(const std::string &key);

        protected:

            // check if the last update and get ready to terminate the combiner
            void check_last();

            // check if can terminate the combiner
            bool terminate();

            // initialize buff
            char* init_buff(size_t thr);

            // get the total size of items in the queue
            size_t get_size();

            // check if combiner is full
            bool full(size_t s);

            bool full();

        private:
            GRAD_LIST combiner;
            char* combiner_buff;
            bool terminate_flag;
            size_t combiner_size;
            size_t updated_size;
            size_t update_size_threshold;
            size_t buff_threshold;
            bool is_full;
            Optimizer optimizer;
            COMM communicator;
            std::unordered_map<std::string, bool> para_flag;
    };

    void update(Dist_opt &opt, int tID, const GRAD_LIST &grads);

    void combine(Dist_opt &opt);

    bool wait(Dist_opt &opt, const std::vector<std::string> &keys);

    bool wait(Dist_opt &opt, const std::string &key);
}

#endif