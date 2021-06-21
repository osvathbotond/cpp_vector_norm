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

template<typename T>
double gpu_lp(double* vec, int vector_length, T p) {
    // Host-side variables
    std::vector <double> pows(vector_length);
    double res;

    size_t bytes = vector_length * sizeof(double);

    // ceil(vector_length / NUM_THREADS)
    int NUM_BLOCKS = (vector_length + NUM_THREADS - 1) / NUM_THREADS;

    // Pointers to the device-side variables
    double* d_vec, * d_res;

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

    // The first sum-reduction. Each block gives back a number, so the first NUM_BLOCKS elements
    // of the result d_res will have the needed information for us (the partial sums).
    sum_reduction_double << <NUM_BLOCKS, NUM_THREADS >> > (d_vec, d_res, vector_length, true, p);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Since a reduction gives us back NUM_BLOCKS elements, we need to do it until NUM_BLOCKS == 1.
    int left;
    int NUM_BLOCKS_RED = NUM_BLOCKS;
    do {
        left = NUM_BLOCKS_RED;
        NUM_BLOCKS_RED = (left + NUM_THREADS - 1) / NUM_THREADS;
        sum_reduction_double << <NUM_BLOCKS_RED, NUM_THREADS >> > (d_res, d_res, left, false, 0);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
            return -1;
        }
    } while (NUM_BLOCKS_RED > 1);

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

    return norm;
}

template<typename T>
double gpu_lp(float* vec, int vector_length, T p) {
    // Host-side variables
    std::vector <float> pows(vector_length);
    float res;

    size_t bytes = vector_length * sizeof(float);

    // ceil(vector_length / NUM_THREADS)
    int NUM_BLOCKS = (vector_length + NUM_THREADS - 1) / NUM_THREADS;

    // Pointers to the device-side variables
    float* d_vec, * d_res;

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

    // The first sum-reduction. Each block gives back a number, so the first NUM_BLOCKS elements
    // of the result d_res will have the needed information for us (the partial sums).
    sum_reduction_float << <NUM_BLOCKS, NUM_THREADS >> > (d_vec, d_res, vector_length, true, p);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Since a reduction gives us back NUM_BLOCKS elements, we need to do it until NUM_BLOCKS == 1.
    int left;
    int NUM_BLOCKS_RED = NUM_BLOCKS;
    do {
        left = NUM_BLOCKS_RED;
        NUM_BLOCKS_RED = (left + NUM_THREADS - 1) / NUM_THREADS;
        sum_reduction_float << <NUM_BLOCKS_RED, NUM_THREADS >> > (d_res, d_res, left, false, 0);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
            return -1;
        }
    } while (NUM_BLOCKS_RED > 1);

    // Copying back to the host (only one number; the 0-th element of the d_res), with error handling.
    err = cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Freeing the memory on the device. Not doing so can cause memory-leak, with error handling.
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

    return norm;
}

int main() {
    const int vector_length = 511*11*17;
    const int p = 1;

    std::vector <float> vec(vector_length);

    // Not the most effective way of creating a random vector.
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-1.0f, 1.0f);
    for (int i = 0; i < vector_length; i++) {
        vec[i] = float(dist(e2));
    }

    // Not the most accurate way of timing the process.
    auto start = std::chrono::steady_clock::now();
    auto res = gpu_lp(vec.data(), vector_length, p);
    auto finish = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();

    std::cout << "Calculating the " << p << "-norm of a(n) " << vector_length << " long random float vector on the GPU took " << elapsed_seconds << " seconds." << std::endl;
    std::cout << "The result is: " << res;

    return 0;

}