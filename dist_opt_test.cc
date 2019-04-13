#include <iostream>
#include <cstdlib>
#include <vector>
#include <utility>
#include <unordered_map>
#include <thread>
#include <string>
#include "dist_opt.h"

#define LEN 6

int N, M;

size_t init(PARA_MAP &model) {
    model.clear();
    size_t cnt = 0;
    for (int i = 0; i < N; i++) {
        std::string key = std::to_string(i + 1);
        DATA_TYPE *val;
        val = new DATA_TYPE[LEN];
        for (int j = 0; j < LEN; j++) {
            val[j] = 0.1 * (i + 1) + 0.01 * (j + 1);
        }
        size_t size = sizeof(DATA_TYPE) * LEN;
        model.insert(std::pair<std::string, PARA_TYPE>(key, PARA_TYPE(val, size)));
        cnt += size;
    }
    return cnt;
}

void clear(PARA_MAP &model) {
    for (auto iter = model.begin(); iter != model.end(); iter++) {
        delete_arr(iter->second.first);
    }
    model.clear();
}

void print(const PARA_MAP &model) {
    printf("=======================================================\nprinting model:\n");
    for (auto iter = model.begin(); iter != model.end(); iter++) {
        std::string key;
        DATA_TYPE *val;
        size_t size;
        PARA_TYPE temp;
        std::tie(key, temp) = *iter;
        std::tie(val, size) = temp;
        printf("    %s (size : %lu) : [", key.c_str(), size);
        for (int i = 0; i * sizeof(DATA_TYPE) < size; i++) {
            printf("%.2f ", val[i]);
        }
        printf("]\n");
    }
    printf("=======================================================\n");
}

size_t init(GRAD_LIST &grads){
    grads.clear();
    size_t cnt = 0;
    for (int i = 0; i < N; i++) {
        std::string key = std::to_string(i + 1);
        DATA_TYPE *val;
        size_t size;
        val = new DATA_TYPE[LEN];
        for (int j = 0; j < LEN; j++) {
            val[j] = 0.1 * (i + 1) + 0.01 * (j + 1);
        }
        size = sizeof(DATA_TYPE) * LEN;
        grads.push_back(std::pair<std::string, PARA_TYPE>(key, PARA_TYPE(val, size)));
        cnt += size;
    }
    return cnt;
}

void clear(GRAD_LIST &grads) {
    for (auto iter = grads.begin(); iter != grads.end(); iter++) {
        delete_arr(iter->second.first);
    }
    grads.clear();
}

int main(int argc, char** argv) {
    N = std::atoi(argv[1]);
    M = std::atoi(argv[2]);
    size_t thr = std::atoi(argv[3]);

    PARA_MAP test_model;
    init(test_model);
    print(test_model);
    
    GRAD_LIST test_grads;
    size_t global_size = init(test_grads);
    global_size *= M;
    printf("global update size = %lu\n", global_size);

    Optimizer test_opt(test_model, 0.5);
    COMM comm = 0;
    dist_opt::Dist_opt test_dist_opt(thr, global_size, comm, test_opt);

    printf("=======================================================\nStart to update!\n");

    std::thread main_thread = std::thread(dist_opt::combine, std::ref(test_dist_opt));

    std::vector<std::thread> threads;
    threads.clear();
    for (int i = 0; i < M; i++) {
        threads.push_back(std::thread(dist_opt::update, std::ref(test_dist_opt), i, test_grads));
        threads.back().join();
    }

    main_thread.join();

    clear(test_grads);
    clear(test_model);

    return 0;
}