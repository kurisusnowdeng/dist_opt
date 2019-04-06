#include "dist_opt.h"
#include <iostream>
#include <thread>
#include <string.h>

#define N 10
#define M 10

int main() {
    Optimizer opt(0.5);
    size_t thr = 128;
    COMM comm = 0;
    DATA_TYPE para[N];
    memset(para, 0, sizeof(para));
    PTR key[N];
    for (int i = 0; i < N; i++) {
        key[i] = &(para[i]);
    }
    DATA_TYPE val[N];
    for (int i = 0; i < N; i++) {
        val[i] = 0.1 * i;
    }
    Dist_opt dist_opt(thr, comm, opt);

    std::thread *t = new std::thread[M];

    for (int i = 0; i < M; i++) {
        t[i] = std::thread(&Dist_opt::update, dist_opt, key, val, N);
    }

    for (int i = 0; i < M; i++) {
        t[i].detach();
        // t[i].join();
    }

    return 0;
}