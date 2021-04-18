#include <vector>
#include <future>
#include <cmath>
#include <fstream>

template<typename T1, typename T2>
T1 lp_sum(std::vector<T1> *vec, T2 p, int i_start, int i_end) {
    T1 sum = 0;
    for (int i = i_start; i < i_end; i++) {
        sum += std::pow(std::abs((*vec)[i]), p);
    }
    return sum;
}

template<typename T1, typename T2>
T1 serial_lp(std::vector<T1> *vec, int n, T2 p) {
    T1 sum = lp_sum(vec, p, 0, n);
    T1 norm = std::pow(sum, 1.0 / p);
    return norm;
}

template<typename T1, typename T2>
T1 parallel_lp(std::vector<T1> *vec, int n, T2 p, int number_of_threads) {
    // If we are using only 1 thread, we should go with serial
    if (number_of_threads == 1) {
        return serial_lp(vec, n, p);
    }

    std::vector<std::future<T1>> futures;

    int delta = n / number_of_threads;
    int i_start = 0;
    int i_end = delta;

    for (int i = 0; i < number_of_threads; i++) {
        if (i == number_of_threads - 1) {
            i_end = n;
        }
        futures.push_back(std::async(std::launch::async,
                                     [](std::vector<T1> *vec, T2 p, int i0, int i1) { return lp_sum(vec, p, i0, i1); },
                                     vec, p, i_start, i_end)); //lp_sum, vec, p, i_start, i_end

        i_start += delta;
        i_end += delta;
    }

    T1 sum = 0;
    for (auto &&fut:futures) {
        sum += fut.get();
    }
    T1 norm = std::pow(sum, 1.0 / p);
    return norm;
}