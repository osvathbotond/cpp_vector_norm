#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <typeinfo>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <numeric>

#include "norms_cpu.cpp"

#define NUM_THREADS 512
#define FLOAT_MEM_SIZE NUM_THREADS * 4
#define DOUBLE_MEM_SIZE NUM_THREADS * 8

template<typename T1, typename T2>
__global__ void lp_pow(const T1* vec, const int n, T2 p, T1* pows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        pows[tid] = pow(abs(vec[tid]), p);
    } 
}

__global__ void sum_reduction_double(double* vec, double* res, const int n) {
    __shared__ double partial_sum[DOUBLE_MEM_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        partial_sum[threadIdx.x] = vec[tid];
    }
    else {
        partial_sum[threadIdx.x] = 0.0;
    }

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        res[blockIdx.x] = partial_sum[0];
    }

}

__global__ void sum_reduction_float(float* vec, float* res, const int n) {
    __shared__ float partial_sum[FLOAT_MEM_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        partial_sum[threadIdx.x] = vec[tid];
    }
    else {
        partial_sum[threadIdx.x] = 0.0f;
    }

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        res[blockIdx.x] = partial_sum[0];
    }

}

template<typename T>
double gpu_lp(double *vec, int vector_length, T p) {
    std::vector <double> pows(vector_length);
    double res;
    
    size_t bytes = vector_length * sizeof(double);

    int NUM_BLOCKS = (vector_length + NUM_THREADS - 1) / NUM_THREADS;

    double* d_vec, * d_pows, * d_res;
    cudaMalloc(&d_vec, bytes);
    cudaMalloc(&d_pows, bytes);
    cudaMalloc(&d_res, bytes);

    cudaMemcpy(d_vec, vec, bytes, cudaMemcpyHostToDevice);

    lp_pow << <NUM_BLOCKS, NUM_THREADS >> > (d_vec, vector_length, p, d_pows);
    sum_reduction_double << <NUM_BLOCKS, NUM_THREADS >> > (d_pows, d_res, vector_length);
    sum_reduction_double << <1, NUM_THREADS >> > (d_res, d_res, vector_length);

    cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_pows);
    cudaFree(d_res);

    double norm = std::pow(res, 1.0 / p);

    return norm;
}

template<typename T>
double gpu_lp(float* vec, int vector_length, T p) {
    std::vector <float> pows(vector_length);
    float res;

    size_t bytes = vector_length * sizeof(float);


    int NUM_BLOCKS = (vector_length + NUM_THREADS - 1) / NUM_THREADS;

    float* d_vec, * d_pows, * d_res;
    cudaMalloc(&d_vec, bytes);
    cudaMalloc(&d_pows, bytes);
    cudaMalloc(&d_res, bytes);

    cudaMemcpy(d_vec, vec, bytes, cudaMemcpyHostToDevice);

    lp_pow << <NUM_BLOCKS, NUM_THREADS >> > (d_vec, vector_length, p, d_pows);
    sum_reduction_float << <NUM_BLOCKS, NUM_THREADS >> > (d_pows, d_res, vector_length);
    sum_reduction_float << <1, NUM_THREADS >> > (d_res, d_res, vector_length);

    cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_pows);
    cudaFree(d_res);

    double norm = std::pow(res, 1.0 / p);

    return norm;
}

template<typename T>
void generate_double_times(int cpu_threads, int total_runs, int vector_length, T p) {
    std::ofstream output;

    std::string fn = std::string("double_p") + std::to_string(p) + std::string("_n") +
        std::to_string(vector_length) + std::string(".dat");
    output.open(fn);
    output << "# The first column contains the cpu time with " << cpu_threads << " threads, and the second column contains the gpu times. \n";
    for (int run = 0; run < total_runs; run++) {
        std::vector <double> vec(vector_length);

        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(0, 25);
        for (int i = 0; i < vector_length; i++) {
            vec[i] = dist(e2);
        }

        auto start_cpu = std::chrono::steady_clock::now();
        auto res_cpu = parallel_lp(&vec, vector_length, p, cpu_threads);
        auto finish_cpu = std::chrono::steady_clock::now();
        double elapsed_seconds_cpu = std::chrono::duration_cast<std::chrono::duration<double>>(finish_cpu - start_cpu).count();

        auto start_gpu = std::chrono::steady_clock::now();
        auto res_gpu = gpu_lp(vec.data(), vector_length, p);
        auto finish_gpu = std::chrono::steady_clock::now();
        double elapsed_seconds_gpu = std::chrono::duration_cast<std::chrono::duration<double>>(finish_gpu - start_gpu).count();

        output << elapsed_seconds_cpu << ", " << elapsed_seconds_gpu << "\n";

        std::cout << "cpu: " << res_cpu << ", gpu: " << res_gpu << "\n";
    }
    output.close();
}

template<typename T>
void generate_float_times(int cpu_threads, int total_runs, int vector_length, T p) {
    std::ofstream output;

    std::string fn = std::string("float_p") + std::to_string(p) + std::string("_n") +
        std::to_string(vector_length) + std::string(".dat");
    output.open(fn);
    output << "# The first column contains the cpu time with " << cpu_threads << " threads, and the second column contains the gpu times. \n";
    for (int run = 0; run < total_runs; run++) {
        std::vector <float> vec(vector_length);

        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(0, 25);
        for (int i = 0; i < vector_length; i++) {
            vec[i] = float(dist(e2));
        }

        auto start_cpu = std::chrono::steady_clock::now();
        auto res_cpu = parallel_lp(&vec, vector_length, p, cpu_threads);
        auto finish_cpu = std::chrono::steady_clock::now();
        double elapsed_seconds_cpu = std::chrono::duration_cast<std::chrono::duration<double>>(finish_cpu - start_cpu).count();

        auto start_gpu = std::chrono::steady_clock::now();
        auto res_gpu = gpu_lp(vec.data(), vector_length, p);
        auto finish_gpu = std::chrono::steady_clock::now();
        double elapsed_seconds_gpu = std::chrono::duration_cast<std::chrono::duration<double>>(finish_gpu - start_gpu).count();

        output << elapsed_seconds_cpu << ", " << elapsed_seconds_gpu << "\n";

        std::cout << "cpu: " << res_cpu << ", gpu: " << res_gpu << "\n";

    }
    output.close();
}

int main() {
    const int cpu_threads = 12;
    const int total_runs = 10;
    const int vector_length = 100000;

    for (int i = 2; i <= 2; i++) {
        generate_double_times(cpu_threads, total_runs, vector_length, i);
        std::cout << "double " << i << " is done. \n";

        //generate_float_times(cpu_threads, total_runs, vector_length, i);
        //std::cout << "float " << i << " is done. \n";
    }

}