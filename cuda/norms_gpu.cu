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
#include <string>

#include "norms_cpu.cpp"

static const int NUM_THREADS = 512;

template<typename T>
__global__ void sum_reduction_double(double* vec, double* res, const int n, const bool power, T p) {
    __shared__ double partial_sum[NUM_THREADS];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (power) {
            partial_sum[threadIdx.x] = pow(abs(vec[tid]), p);
        }
        else {
            partial_sum[threadIdx.x] = vec[tid];
        }
    }
    else {
        partial_sum[threadIdx.x] = 0.0;
    }

    // Sync the threads to have all of the needed data in the shared memory
    __syncthreads();

    // Do the reduction (example with 8 numbers):
    // a               b       c   d   e f g h
    // a+e             b+f     c+g d+h e f g h
    // a+e+c+g         b+f+c+g c+g d+h e f g h
    // a+e+c+g+b+f+c+g b+f+c+g c+g d+h e f g h
    // And the result is just the 0-th element
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        // We do need to wait for all of the threads to do the sums
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        res[blockIdx.x] = partial_sum[0];
    }

}

template<typename t>
__global__ void sum_reduction_float(float* vec, float* res, const int n, const bool power, t p) {
    __shared__ float partial_sum[NUM_THREADS];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (power) {
            partial_sum[threadIdx.x] = pow(abs(vec[tid]), p);
        }
        else {
            partial_sum[threadIdx.x] = vec[tid];
        }
    }
    else {
        partial_sum[threadIdx.x] = 0;
    }

    // Sync the threads to have all of the needed data in the shared memory
    __syncthreads();

    // Do the reduction (example with 8 numbers):
    // a               b       c   d   e f g h
    // a+e             b+f     c+g d+h e f g h
    // a+e+c+g         b+f+c+g c+g d+h e f g h
    // a+e+c+g+b+f+c+g b+f+c+g c+g d+h e f g h
    // And the result is just the 0-th element
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        // We do need to wait for all of the threads to do the sums
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        res[blockIdx.x] = partial_sum[0];
    }

}
// float, because it returns the time and not the result (it's double on the main branch)
template<typename T>
float gpu_lp(double* vec, int vector_length, T p) {
    // Host-side variables
    std::vector <double> pows(vector_length);
    double res;

    size_t bytes = vector_length * sizeof(double);

    // ceil(vector_length / NUM_THREADS)
    int NUM_BLOCKS = (vector_length + NUM_THREADS - 1) / NUM_THREADS;

    // Pointers to the device-side variables
    double* d_vec, * d_res;

    // Cuda event for the device-side timing
    cudaEvent_t evt[2];
    for (auto& e : evt) {
        cudaEventCreate(&e);
    }

    // Allocate the memory on the GPU and move the vector (with error handling)
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_vec, bytes);
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

    // The first event, to time only the calculations
    cudaEventRecord(evt[0]);

    // The first sum-reduction. Each block gives back a number, so the first NUM_BLOCKS elements
    // of the result d_res will have the needed information for us (the partial sums).
    sum_reduction_double << <NUM_BLOCKS, NUM_THREADS >> > (d_vec, d_res, vector_length, true, p);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Since a reduction gives us back NUM_BLOCKS elements, we need to do it until NUM_BLOCKS == 1.
    int left = (int)std::ceil(vector_length / (1.0 * NUM_THREADS));
    int NUM_BLOCKS_RED = (int)std::ceil(NUM_BLOCKS / (1.0 * NUM_THREADS));
    while (left > 1) {
        sum_reduction_double << <NUM_BLOCKS_RED, NUM_THREADS >> > (d_res, d_res, left, false, 0);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
            return -1;
        }
        left = (int)std::ceil(left / (1.0 * NUM_THREADS));
        NUM_BLOCKS_RED = (int)std::ceil(NUM_BLOCKS_RED / (1.0 * NUM_THREADS));
    }
    // The second event, to time only the calculations
    cudaEventRecord(evt[1]);

    // Copying back to the host (only one number; the 0-th element of the d_res), with error handling.
    err = cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Freeing the memory on the device. Not doing so can cause memory-leak.
    err = cudaFree(d_vec);
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

    // Wait for the event evt[1]. This is redundant.
    cudaEventSynchronize(evt[1]);

    //Calculating the time
    float dt = 0.0f;
    cudaEventElapsedTime(&dt, evt[0], evt[1]);

    return dt;
}
// float, because it returns the time and not the result (it's double on the main branch)
template<typename T>
float gpu_lp(float* vec, int vector_length, T p) {
    // Host-side variables
    std::vector <float> pows(vector_length);
    float res;

    size_t bytes = vector_length * sizeof(float);

    // ceil(vector_length / NUM_THREADS)
    int NUM_BLOCKS = (vector_length + NUM_THREADS - 1) / NUM_THREADS;

    // Pointers to the device-side variables
    float* d_vec, * d_res;

    // Cuda event for the device-side timing
    cudaEvent_t evt[2];
    for (auto& e : evt) {
        cudaEventCreate(&e);
    }

    // Allocate the memory on the GPU and move the vector (with error handling)
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_vec, bytes);
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

    // The first event, to time only the calculations
    cudaEventRecord(evt[0]);

    // The first sum-reduction. Each block gives back a number, so the first NUM_BLOCKS elements
    // of the result d_res will have the needed information for us (the partial sums).
    sum_reduction_float << <NUM_BLOCKS, NUM_THREADS >> > (d_vec, d_res, vector_length, true, p);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Since a reduction gives us back NUM_BLOCKS elements, we need to do it until NUM_BLOCKS == 1.
    int left = (int)std::ceil(vector_length / (1.0 * NUM_THREADS));
    int NUM_BLOCKS_RED = (int)std::ceil(NUM_BLOCKS / (1.0 * NUM_THREADS));
    while (left > 1) {
        sum_reduction_float<<<NUM_BLOCKS_RED, NUM_THREADS>>>(d_res, d_res, left, false, 0);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
            return -1;
        }
        left = (int)std::ceil(left / (1.0 * NUM_THREADS));
        NUM_BLOCKS_RED = (int)std::ceil(NUM_BLOCKS_RED / (1.0 * NUM_THREADS));
    }
    // The second event, to time only the calculations
    cudaEventRecord(evt[1]);

    // Copying back to the host (only one number; the 0-th element of the d_res), with error handling.
    err = cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Freeing the memory on the device. Not doing so can cause memory-leak.
    err = cudaFree(d_vec);
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

    // Wait for the event evt[1]. This is redundant.
    cudaEventSynchronize(evt[1]);

    //Calculating the time
    float dt = 0.0f;
    cudaEventElapsedTime(&dt, evt[0], evt[1]);

    return dt;
}

template<typename T>
void generate_double_times(int cpu_threads, int total_runs, int vector_length, T p) {
    // Writing to a text file
    std::ofstream output;

    std::string fn = std::string("double_p") + std::to_string(p) + std::string("_n") +
        std::to_string(vector_length) + std::string(".dat");
    output.open(fn);
    output << "# The first column contains only the calculation times (power and sum done on the cpu) with " << cpu_threads << "threads, the second "
           << "column contains the total cpu time with the same threads, the third column contains the total gpu times (with copy, malloc, etc) and the "
           <<"fourth column contains only the calculation times (power and sum done on the gpu) on the gpu. \n";
    for (int run = 0; run < total_runs; run++) {
        std::vector <double> vec(vector_length);

        // Not the most effective way of creating a random vector.
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-1.0, 1.0);
        for (int i = 0; i < vector_length; i++) {
            vec[i] = dist(e2);
        }

        // Calculating the total times, on the host-side
        auto start_cpu = std::chrono::steady_clock::now();
        auto elapsed_seconds_cpu_calc = parallel_lp(vec, vector_length, p, cpu_threads);
        auto finish_cpu = std::chrono::steady_clock::now();
        double elapsed_seconds_cpu_total = std::chrono::duration_cast<std::chrono::duration<double>>(finish_cpu - start_cpu).count();

        auto start_gpu = std::chrono::steady_clock::now();
        auto res_gpu = gpu_lp(vec.data(), vector_length, p);
        auto finish_gpu = std::chrono::steady_clock::now();
        double elapsed_seconds_gpu_calc = res_gpu / 1000.0;
        double elapsed_seconds_gpu_total = std::chrono::duration_cast<std::chrono::duration<double>>(finish_gpu - start_gpu).count();


        output << elapsed_seconds_cpu_calc << ", " << elapsed_seconds_cpu_total << ", " << elapsed_seconds_gpu_total << ", " << elapsed_seconds_gpu_calc << "\n";
    }
    output.close();
}

template<typename T>
void generate_float_times(int cpu_threads, int total_runs, int vector_length, T p) {
    // Writing to a text file
    std::ofstream output;

    std::string fn = std::string("float_p") + std::to_string(p) + std::string("_n") +
        std::to_string(vector_length) + std::string(".dat");
    output.open(fn);
    output << "# The first column contains only the calculation times (power and sum done on the cpu) with " << cpu_threads << "threads, the second "
           << "column contains the total cpu time with the same threads, the third column contains the total gpu times (with copy, malloc, etc) and the "
           << "fourth column contains only the calculation times (power and sum done on the gpu) on the gpu. \n";
    for (int run = 0; run < total_runs; run++) {
        std::vector <float> vec(vector_length);

        // Not the most effective way of creating a random vector.
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-1.0f, 1.0f);
        for (int i = 0; i < vector_length; i++) {
            vec[i] = float(dist(e2));
        }

        // Calculating the total times, on the host-side
        auto start_cpu = std::chrono::steady_clock::now();
        auto elapsed_seconds_cpu_calc = parallel_lp(vec, vector_length, p, cpu_threads);
        auto finish_cpu = std::chrono::steady_clock::now();
        double elapsed_seconds_cpu_total = std::chrono::duration_cast<std::chrono::duration<double>>(finish_cpu - start_cpu).count();

        auto start_gpu = std::chrono::steady_clock::now();
        auto res_gpu = gpu_lp(vec.data(), vector_length, p);
        auto finish_gpu = std::chrono::steady_clock::now();
        double elapsed_seconds_gpu_calc = res_gpu / 1000.0;
        double elapsed_seconds_gpu_total = std::chrono::duration_cast<std::chrono::duration<double>>(finish_gpu - start_gpu).count();


        output << elapsed_seconds_cpu_calc <<", " << elapsed_seconds_cpu_total << ", " << elapsed_seconds_gpu_total << ", " << elapsed_seconds_gpu_calc << "\n";
    }
    output.close();
}

int main() {
    const int cpu_threads = 12;
    const int total_runs = 100;
    const int vector_length = 10'000'000;

    for (int i = 1; i <= 5; i++) {
        generate_double_times(cpu_threads, total_runs, vector_length, i);
        std::cout << "double " << i << " is done. \n";

        generate_float_times(cpu_threads, total_runs, vector_length, i);
        std::cout << "float " << i << " is done. \n";
    }

}