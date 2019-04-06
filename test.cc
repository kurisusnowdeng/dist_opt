#include<iostream>
#include<tuple>
#include<deque>

// using namespace std;

typedef int (*FUN)(int, int);
typedef float DATA_TYPE;
typedef int COMM;
typedef float* PTR;

// int add(int a, int b) {
//     return a+b;
// }

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

int main() {
    // FUN fun = add;
    // cout << fun(1, 2) << endl;
    // std::deque<std::tuple<int, int>> test;
    // test.clear();
    // for (int i = 0; i < 10; i++) {
    //     test.push_back(std::make_tuple(i, i*i));
    // }
    // while (!test.empty()) {
    //     int a = std::get<0>(test.front());
    //     int b = std::get<1>(test.front());
    //     std::cout << a << ' ' << b << std::endl;
    //     test.pop_front();
    // }
    Optimizer a(0.5);
    Optimizer b = a;
    std::cout << b.get_lr() << std::endl;

    return 0;
}