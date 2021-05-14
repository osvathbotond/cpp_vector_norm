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
float gpu_lp(double* vec, int vector_length, T p) {
    std::vector <double> pows(vector_length);
    double res;

    size_t bytes = vector_length * sizeof(double);

    int NUM_BLOCKS = (vector_length + NUM_THREADS - 1) / NUM_THREADS;

    double* d_vec, * d_pows, * d_res;

    cudaEvent_t evt[2];
    for (auto& e : evt) {
        cudaEventCreate(&e);
    }

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_vec, bytes);
    if (err != cudaSuccess) {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMalloc(&d_pows, bytes);
    if (err != cudaSuccess) {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMalloc(&d_res, bytes);;
    if (err != cudaSuccess) {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMemcpy(d_vec, vec, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    cudaEventRecord(evt[0]);
    lp_pow << <NUM_BLOCKS, NUM_THREADS >> > (d_vec, vector_length, p, d_pows);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during raising to power): " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    sum_reduction_double << <NUM_BLOCKS, NUM_THREADS >> > (d_pows, d_res, vector_length);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    int left = (int)std::ceil(vector_length / (1.0 * NUM_THREADS));
    int NUM_BLOCKS_RED = (int)std::ceil(NUM_BLOCKS / (1.0 * NUM_THREADS));
    while (left > 1) {
        sum_reduction_double << <NUM_BLOCKS_RED, NUM_THREADS >> > (d_res, d_res, left);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
            return -1;
        }
        left = (int)std::ceil(left / (1.0 * NUM_THREADS));
        NUM_BLOCKS_RED = (int)std::ceil(NUM_BLOCKS_RED / (1.0 * NUM_THREADS));
    }
    cudaEventRecord(evt[1]);

    err = cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaFree(d_vec);
    if (err != cudaSuccess) {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaFree(d_pows);
    if (err != cudaSuccess) {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaFree(d_res);
    if (err != cudaSuccess) {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    double norm = std::pow(res, 1.0 / p);

    cudaEventSynchronize(evt[1]);
    float dt = 0.0f;
    cudaEventElapsedTime(&dt, evt[0], evt[1]);

    return dt;
}

template<typename T>
float gpu_lp(float* vec, int vector_length, T p) {
    std::vector <float> pows(vector_length);
    float res;

    size_t bytes = vector_length * sizeof(float);

    int NUM_BLOCKS = (vector_length + NUM_THREADS - 1) / NUM_THREADS;

    float* d_vec, * d_pows, * d_res;

    cudaEvent_t evt[2];
    for (auto& e : evt) {
        cudaEventCreate(&e);
    }

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_vec, bytes);
    if (err != cudaSuccess) {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMalloc(&d_pows, bytes);
    if (err != cudaSuccess) {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMalloc(&d_res, bytes);;
    if (err != cudaSuccess) {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMemcpy(d_vec, vec, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    cudaEventRecord(evt[0]);
    lp_pow << <NUM_BLOCKS, NUM_THREADS >> > (d_vec, vector_length, p, d_pows);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during raising to power): " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    sum_reduction_float << <NUM_BLOCKS, NUM_THREADS >> > (d_pows, d_res, vector_length);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    int left = (int)std::ceil(vector_length / (1.0 * NUM_THREADS));
    int NUM_BLOCKS_RED = (int)std::ceil(NUM_BLOCKS / (1.0 * NUM_THREADS));
    while (left > 1) {
        sum_reduction_float<<<NUM_BLOCKS_RED, NUM_THREADS>>>(d_res, d_res, left);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
            return -1;
        }
        left = (int)std::ceil(left / (1.0 * NUM_THREADS));
        NUM_BLOCKS_RED = (int)std::ceil(NUM_BLOCKS_RED / (1.0 * NUM_THREADS));
    }
    cudaEventRecord(evt[1]);

    err = cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaFree(d_vec);
    if (err != cudaSuccess) {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaFree(d_pows);
    if (err != cudaSuccess) {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaFree(d_res);
    if (err != cudaSuccess) {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    double norm = std::pow(res, 1.0 / p);

    cudaEventSynchronize(evt[1]);
    float dt = 0.0f;
    cudaEventElapsedTime(&dt, evt[0], evt[1]);

    return dt;
}

template<typename T>
void generate_double_times(int cpu_threads, int total_runs, int vector_length, T p) {
    std::ofstream output;

    std::string fn = std::string("double_p") + std::to_string(p) + std::string("_n") +
        std::to_string(vector_length) + std::string(".dat");
    output.open(fn);
    output << "# The first column contains the cpu time with " << cpu_threads << " threads, and the second column contains the total gpu times (with copy, malloc, etc) and the third contains"
        << " only the calculation times (power and sum done on the gpu). \n";
    for (int run = 0; run < total_runs; run++) {
        std::vector <double> vec(vector_length);

        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-1000.0, 1000.0);
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
        double elapsed_seconds_gpu_calc = res_gpu / 1000.0;
        double elapsed_seconds_gpu_total = std::chrono::duration_cast<std::chrono::duration<double>>(finish_gpu - start_gpu).count();


        output << elapsed_seconds_cpu << ", " << elapsed_seconds_gpu_total << ", " << elapsed_seconds_gpu_calc << "\n";
    }
    output.close();
}

template<typename T>
void generate_float_times(int cpu_threads, int total_runs, int vector_length, T p) {
    std::ofstream output;

    std::string fn = std::string("float_p") + std::to_string(p) + std::string("_n") +
        std::to_string(vector_length) + std::string(".dat");
    output.open(fn);
    output << "# The first column contains the cpu time with " << cpu_threads << " threads, and the second column contains the total gpu times (with copy, malloc, etc) and the third contains"
           <<" only the calculation times (power and sum done on the gpu). \n";
    for (int run = 0; run < total_runs; run++) {
        std::vector <float> vec(vector_length);

        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-1000.0f, 1000.0f);
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
        double elapsed_seconds_gpu_calc = res_gpu / 1000.0;
        double elapsed_seconds_gpu_total = std::chrono::duration_cast<std::chrono::duration<double>>(finish_gpu - start_gpu).count();


        output << elapsed_seconds_cpu << ", " << elapsed_seconds_gpu_total << ", " << elapsed_seconds_gpu_calc << "\n";
    }
    output.close();
}

int main() {
    const int cpu_threads = 12;
    const int total_runs = 15;
    const int vector_length = 10'000'000;

    for (int i = 1; i <= 5; i++) {
        generate_double_times(cpu_threads, total_runs, vector_length, i);
        std::cout << "double " << i << " is done. \n";

        generate_float_times(cpu_threads, total_runs, vector_length, i);
        std::cout << "float " << i << " is done. \n";
    }

}