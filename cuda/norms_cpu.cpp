#include <vector>
#include <future>
#include <cmath>

template<typename T>
double lp_sum(const std::vector<T>& vec, T p, int i_start, int i_end) {
    double sum = 0;
    for (int i = i_start; i < i_end; i++) {
        sum += std::pow(std::abs(vec[i]), p);
    }
    return sum;
}

template<typename T>
double serial_lp(const std::vector<T>& vec, int n, T p) {
    double sum = lp_sum(vec, p, 0, n);
    double norm = std::pow(sum, 1.0 / p);
    return norm;
}

template<typename T>
double parallel_lp(const std::vector<T>& vec, int n, T p, int number_of_threads) {
    // If we are using only 1 thread, we should go with serial
    if (number_of_threads == 1) {
        auto start_cpu = std::chrono::high_resolution_clock::now();
        auto norm = serial_lp(vec, n, p);
        auto finish_cpu = std::chrono::high_resolution_clock::now();
        double elapsed_seconds_cpu = std::chrono::duration_cast<std::chrono::microseconds>(finish_cpu - start_cpu).count() * 0.001;
        return elapsed_seconds_cpu;

        // The return removed; we are returning the calculation time instead
        // return norm;
    }

    std::vector<std::future<double>> futures;
    futures.reserve(number_of_threads);

    int delta = n / number_of_threads;
    int i_start = 0;
    int i_end = delta;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < number_of_threads; i++) {
        if (i == number_of_threads - 1) {
            i_end = n;
        }
        futures.push_back(std::async(std::launch::async,
            [](const std::vector<T>& vec, T p, int i0, int i1) { return lp_sum(vec, p, i0, i1); },
            std::cref(vec), p, i_start, i_end)); //lp_sum, vec, p, i_start, i_end

        i_start += delta;
        i_end += delta;
    }

    double sum = 0;
    for (auto&& fut : futures) {
        sum += fut.get();
    }
    double norm = std::pow(sum, 1.0 / p);

    auto finish_cpu = std::chrono::high_resolution_clock::now();
    double elapsed_seconds_cpu = std::chrono::duration_cast<std::chrono::microseconds>(finish_cpu - start_cpu).count() * 0.001;

    return elapsed_seconds_cpu;

    // The return removed; we are returning the calculation time instead
    // return norm;
}