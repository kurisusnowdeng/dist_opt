#include <iostream>
#include <vector>
#include <utility>
#include <unordered_map>
#include <thread>
#include <string>
#include "dist_opt.h"

#define N 10
#define M 10

int main() {
    PARA_MAP test_model;
    test_model.clear();
    for (int i = 0; i < N; i++) {
        test_model.insert(std::make_pair<std::string, DATA_TYPE>(std::to_string(i + 1), 0.1 * (i + 1)));
    }
    // for (auto iter = test_model.begin(); iter != test_model.end(); iter++) {
    //     printf("%s : %.3f\n", iter->first.c_str(), iter->second);
    // }

    Optimizer test_opt(test_model, 0.5);
    size_t thr = 128;
    COMM comm = 0;
    Dist_opt test_dist_opt(thr, comm, test_opt);
    
    GRAD_LIST test_grads;
    test_grads.clear();
    for (int i = 0; i < N; i++) {
        test_grads.push_back(std::make_pair<std::string, DATA_TYPE>(std::to_string(i + 1), 0.1 * (i + 1)));
    }

    std::thread *t = new std::thread[M];

    for (int i = 0; i < M; i++) {
        t[i] = std::thread(&Dist_opt::update, test_dist_opt, i, test_grads);
    }

    for (int i = 0; i < M; i++) {
        t[i].detach();
        // t[i].join();
    }

    delete [] t;

    return 0;
}