#include <chrono>
#include "dist_opt.h"

namespace dist_opt {

    std::mutex mtx;
    std::condition_variable pop_all;
    std::condition_variable push;

    inline void data_transfer(const void *src, void *dst, size_t n) {
        memcpy(dst, src, n);
    }

    char* Dist_opt::init_buff(size_t thr) {
        return (new char[thr]);
    }

    size_t Dist_opt::get_size() {
        return combiner_size;
    }

    bool Dist_opt::full(size_t s) {
        if (get_size() + s >= buff_threshold) {
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));
            is_full = true;
            printf("Combiner is full!\n");
        }
        return is_full;
    }

    bool Dist_opt::full() {
        return is_full;
    }

    void Dist_opt::check_last() {
        printf("updated size : %lu\n", updated_size);
        if (updated_size >= update_size_threshold) {
            terminate_flag = true;
            pop_all.notify_one();
            printf("No more to update!\n");
        }
    }

    bool Dist_opt::terminate() {
        return terminate_flag;
    }

    Dist_opt::Dist_opt(size_t threshold, size_t size, const COMM &comm, const Optimizer &opt) {
        buff_threshold = threshold;
        combiner_buff = init_buff(buff_threshold);
        combiner.clear();
        para_flag.clear();
        combiner_size = 0;
        updated_size = 0;
        is_full = false;
        update_size_threshold = size;
        communicator = comm;
        optimizer = opt;
    }

    Dist_opt::~Dist_opt() {
        delete_arr(combiner_buff);
    }

    void Dist_opt::update(int tID, const GRAD_LIST &grads) {
        std::string key;
        DATA_TYPE *val;
        size_t size;
        PARA_TYPE temp;
        for (auto item: grads) {
            std::tie(key, temp) = item;
            std::tie(val, size) = temp;
            std::unique_lock<std::mutex> locker(mtx);
            while (full(size)) {
                printf("Thread %d: update %s waiting...\n", tID, key.c_str());
                pop_all.notify_one();
                push.wait(locker);
            }
            combiner.push_back(item);
            combiner_size += size;
            updated_size += size;
            printf("Thread %d: update %s done. Current size = %lu.\n", tID, key.c_str(), get_size());
            check_last();
            locker.unlock();
        }
    }

    void Dist_opt::combine() {
        printf("Combiner start!\n");
        terminate_flag = false;
        while (!terminate()) {
            std::unique_lock<std::mutex> locker(mtx);
            while (!full() && !terminate()) {
                printf("Combiner waiting for update...\n");
                pop_all.wait(locker);
            }
            printf("Start to pop all...\n");

            // // memcpy to buff
            // size_t buff_size = 0;
            // for (auto iter = combiner.begin(); iter != combiner.end(); iter++) {
            //     std::string key;
            //     PARA_TYPE val;
            //     std::tie(key, val) = *iter;
            //     data_transfer(val.first, combiner_buff + buff_size, val.second);
            //     buff_size += val.second;
            // }
            // // call do_allreduce

            // // memcpy from buff and update
            // buff_size = 0;
            // for (auto iter = combiner.begin(); iter != combiner.end(); iter++) {
            //     std::string key;
            //     PARA_TYPE val;
            //     std::tie(key, val) = *iter;
            //     data_transfer(combiner_buff + buff_size, val.first, val.second);
            //     buff_size += val.second;
            // }

            // update parameters
            optimizer.update(combiner);
            combiner.clear();
            combiner_size = 0;
            is_full = false;

            printf("Combiner update done!\n");
            locker.unlock();
            push.notify_all();
        }
    }

    bool Dist_opt::wait(const std::vector<std::string> &keys) {
        return true;
    }

    bool Dist_opt::wait(const std::string &key) {
        return true;
    }

    void update(Dist_opt &opt, int tID, const GRAD_LIST &grads) {
        opt.update(tID, grads);
    }

    void combine(Dist_opt &opt) {
        opt.combine();
    }

    bool wait(Dist_opt &opt, const std::vector<std::string> &keys) {
        return opt.wait(keys);
    }

    bool wait(Dist_opt &opt, const std::string &key) {
        return opt.wait(key);
    }

}