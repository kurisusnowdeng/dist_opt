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

Dist_opt::Dist_opt(size_t thr, COMM &comm, Optimizer &opt) {
    threshold = thr;
    combiner_buff = init_buff(threshold);
    combiner.clear();
    size = 0;
    communicator = comm;
    optimizer = opt;
    combine();
}

// Dist_opt::~Dist_opt() {
//     delete [] combiner_buff;
// }

void Dist_opt::update(PTR keys[], DATA_TYPE grads[], int num) {
    for (int i = 0; i < num; i++) {
        while (full(sizeof(grads[i]))) {
            pop_all.notify_one();
            printf("Update %d waiting...", i);
            push.wait();
        }
        mtx.lock();
        combiner.push_back(std::make_pair(keys[i], grads[i]));
        size += sizeof(grads[i]);
        printf("Update %d done.", i);
        mtx.unlock();
    }
}

void Dist_opt::combine() {
    while (!full(0)) {
        printf("Combiner waiting...");
        pop_all.wait();
    }
    mtx.lock();
    // test pop_all
    DATA_TYPE sum = 0;
    while (!combiner.empty()) {
        DATA_TYPE item = std::get<0>(combiner.front());
        sum += item;
        printf("item = %.3f, sum = %.3f\n", item, sum);
        combiner.pop_front();
    }
    size = 0;
    printf("Combiner done!");
    mtx.unlock();
    push.notify_one();
    // memcpy to buff

    // call do_allreduce

    // memcpy from buff

    // update parameters

    printf("Combiner update done!");
}

// bool Dist_opt::wait(PTR keys[]) {

// }