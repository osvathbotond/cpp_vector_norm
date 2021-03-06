#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <typeinfo>
#include <string>
#include "norms_cpu.cpp"

template<typename T>
void generate_double_times(int min_threads, int max_threads, int total_runs, int vector_length, T p) {
    std::ofstream output;
    std::string fn = std::string("double_p") + std::to_string(p) + std::string("_n") +
                     std::to_string(vector_length) + std::string(".dat");
    output.open(fn);
    output << "# Each column contains the run times for a given thread. They are, in order: ";
    for (int thread = min_threads; thread <= max_threads; thread++) {
        if (thread == max_threads) {
            output << thread << ".\n";
        } else {
            output << thread << ", ";
        }
    }

    for (int run = 0; run < total_runs; run++) {
        std::vector <double> vec(vector_length);

        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-1.0, 1.0);
        for (int i = 0; i < vector_length; i++) {
            vec[i] = dist(e2);
        }
        for (int thread = min_threads; thread <= max_threads; thread++) {
            auto start = std::chrono::steady_clock::now();
            auto res = parallel_lp(vec, vector_length, (double)p, thread);
            auto finish = std::chrono::steady_clock::now();
            double elapsed_seconds = std::chrono::duration_cast<
                    std::chrono::duration<double> >(finish - start).count();
            if (thread == max_threads) {
                output << res << "\n";
            } else {
                output << res << ", ";
            }
        }
    }
    output.close();
}

template<typename T>
void generate_float_times(int min_threads, int max_threads, int total_runs, int vector_length, T p) {
    std::ofstream output;
    std::string fn = std::string("float_p") + std::to_string(p) + std::string("_n") +
                     std::to_string(vector_length) + std::string(".dat");
    output.open(fn);
    output << "# Each column contains the run times for a given thread. They are, in order: ";
    for (int thread = min_threads; thread <= max_threads; thread++) {
        if (thread == max_threads) {
            output << thread << ".\n";
        } else {
            output << thread << ", ";
        }
    }
    for (int run = 0; run < total_runs; run++) {
        std::vector <float> vec(vector_length);

        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-1.0f, 1.0f);
        for (int i = 0; i < vector_length; i++) {
            vec[i] = float(dist(e2));
        }

        for (int thread = min_threads; thread <= max_threads; thread++) {
            auto start = std::chrono::steady_clock::now();
            auto res = parallel_lp(vec, vector_length, (float)p, thread);
            auto finish = std::chrono::steady_clock::now();
            double elapsed_seconds = std::chrono::duration_cast<
                    std::chrono::duration<double> >(finish - start).count();
            if (thread == max_threads) {
                output << res << "\n";
            } else {
                output << res << ", ";
            }
        }
    }
    output.close();
}


int main() {
    const int min_threads = 1;
    const int max_threads = 12;
    const int total_runs = 100;
    const int vector_length = 1'000'000;
    for (int i = 1; i <= 5; i ++) {
        generate_double_times(min_threads, max_threads, total_runs, vector_length, i);
        std::cout << "double " << i << " is done. \n";

        generate_float_times(min_threads, max_threads, total_runs, vector_length, i);
        std::cout << "float " << i << " is done. \n";
    }

}
