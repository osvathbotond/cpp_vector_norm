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
__global__ void sum_reduction_float(float* vec, float* res, const int n, const bool power, T p) {
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

template<typename T>
double gpu_lp(double* vec, int vector_length, T p) {
    std::vector <double> pows(vector_length);
    double res;

    size_t bytes = vector_length * sizeof(double);

    int NUM_BLOCKS = (vector_length + NUM_THREADS - 1) / NUM_THREADS;

    double* d_vec, * d_res;

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
    sum_reduction_double << <NUM_BLOCKS, NUM_THREADS >> > (d_vec, d_res, vector_length, true, p);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
        return -1;
    }
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
    std::vector <float> pows(vector_length);
    float res;

    size_t bytes = vector_length * sizeof(float);

    int NUM_BLOCKS = (vector_length + NUM_THREADS - 1) / NUM_THREADS;

    float* d_vec, * d_res;

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

    sum_reduction_float << <NUM_BLOCKS, NUM_THREADS >> > (d_vec, d_res, vector_length, true, p);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
        return -1;
    }
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

    err = cudaFree(d_res);
    if (err != cudaSuccess) {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    double norm = std::pow(res, 1.0 / p);

    return norm;
}

int main() {
    const int vector_length = 10'000'000;
    const int p = 2;

    std::vector <float> vec(vector_length);

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-1.0f, 1.0f);
    for (int i = 0; i < vector_length; i++) {
        vec[i] = float(dist(e2));
    }

    auto start = std::chrono::steady_clock::now();
    auto res = gpu_lp(vec.data(), vector_length, p);
    auto finish = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();

    std::cout << "Calculating the " << p << "-norm of a(n) " << vector_length << " long random float vector on the GPU took " << elapsed_seconds << " seconds.";

    return 0;

}