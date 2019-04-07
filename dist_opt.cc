#include "dist_opt.h"

DATA_TYPE* Dist_opt::init_buff(size_t thr) {
    return (new DATA_TYPE[thr]);
}

size_t Dist_opt::get_size() {
    return size;
}

bool Dist_opt::full(size_t s) {
    return (get_size() + s >= threshold);
}

Dist_opt::Dist_opt(size_t thr, const COMM &comm, const Optimizer &opt) {
    threshold = thr;
    combiner_buff = init_buff(threshold);
    combiner.clear();
    dict.clear();
    size = 0;
    communicator = comm;
    optimizer = opt;
    ctx = new Context;
    combiner_thread = new std::thread(&Dist_opt::combine, this);
    (*combiner_thread).detach();
    // (*combiner_thread).join();
}

Dist_opt::~Dist_opt() {
    delete [] combiner_buff;
    delete ctx;
    delete combiner_thread;
}

void Dist_opt::update(int id, const GRAD_LIST &grads) {
    std::string key;
    DATA_TYPE val;
    for (auto item: grads) {
        std::tie(key, val) = item;
        std::unique_lock<std::mutex> locker(ctx->mtx);
        while (full(sizeof(val))) {
            ctx->pop_all.notify_one();
            printf("Thread %d: update %s waiting...\n", id, key.c_str());
            ctx->push.wait(locker);
        }
        // ctx->mtx.lock();
        combiner.push_back(item);
        size += sizeof(val);
        printf("Thread %d: update %s done. Current size = %d.\n", id, key.c_str(), get_size());
        locker.unlock();
        // ctx->mtx.unlock();
    }
}

void Dist_opt::combine() {
    printf("Combiner start!\n");
    while (true) {
        std::unique_lock<std::mutex> locker(ctx->mtx);
        while (!full(0)) {
            printf("Combiner waiting...\n");
            ctx->pop_all.wait(locker);
        }
        printf("Combiner full! Start to pop all...");
        // test pop_all
        // ctx->mtx.lock();
        DATA_TYPE sum = 0;
        std::string key;
        DATA_TYPE val;
        while (!combiner.empty()) {
            auto item = combiner.front();
            std::tie(key, val) = item;
            sum += val;
            printf("key = %s, val = %.3f, sum = %.3f\n", key.c_str(), val, sum);
            combiner.pop_front();
        }
        size = 0;
        printf("Combiner done!\n");
        // ctx->mtx.unlock();
        locker.unlock();
        ctx->push.notify_one();
        // memcpy to buff

        // call do_allreduce

        // memcpy from buff

        // update parameters

        printf("Combiner update done!");
    }
}

bool Dist_opt::wait(const std::vector<std::string> &keys) {
    return true;
}

bool Dist_opt::wait(const std::string &key) {
    return true;
}