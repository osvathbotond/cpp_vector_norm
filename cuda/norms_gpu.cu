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

static const int num_threads = 512;

template<typename T>
__global__ void sum_reduction(const T* vec, T* res, const int n, const bool power, T p) {
    __shared__ T partial_sum[num_threads];

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
        if (gridDim.x == 1) {
            res[blockIdx.x] = pow(partial_sum[0], (T)(1.0/p));
        }
        else {
            res[blockIdx.x] = partial_sum[0];
        }
        
    }

}

template<typename T>
T gpu_lp(T* vec, int vector_length, T p) {
    // Host-side variables
    std::vector <T> pows(vector_length);
    T res;

    size_t bytes = vector_length * sizeof(T);

    // ceil(vector_length / num_threads)
    int num_blocks = (vector_length + num_threads - 1) / num_threads;

    // Pointers to the device-side variables
    T *d_vec, *d_res1, *d_res2;

    // Allocate the memory on the GPU and move the vector (with error handling)
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_vec, bytes);
    if (err != cudaSuccess) {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMalloc(&d_res1, bytes);
    if (err != cudaSuccess) {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMalloc(&d_res2, bytes);
    if (err != cudaSuccess) {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMemcpy(d_vec, vec, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // The first sum-reduction. Each block gives back a number, so the first num_blocks elements
    // of the result d_res will have the needed information for us (the partial sums).
    sum_reduction<<<num_blocks, num_threads>>>(d_vec, d_res1, vector_length, true, p);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Since a reduction gives us back num_blocks elements, we need to do it until num_blocks == 1.
    int left;
    int num_blocks_red = num_blocks;
    int source_counter = 1;
    do {
        left = num_blocks_red;
        num_blocks_red = (left + num_threads - 1) / num_threads;
        if (source_counter == 1) {
            sum_reduction<<<num_blocks_red, num_threads>>>(d_res1, d_res2, left, false, p);
            source_counter = 2;
        }
        else {
            sum_reduction<<<num_blocks_red, num_threads>>>(d_res2, d_res1, left, false, p);
            source_counter = 1;
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error in kernel call (during sum reduction): " << cudaGetErrorString(err) << "\n";
            return -1;
        }
    } while (num_blocks_red > 1);

    // Copying back to the host (only one number; the 0-th element of the d_res), with error handling.
    if (source_counter == 1) {
        err = cudaMemcpy(&res, d_res1, sizeof(T), cudaMemcpyDeviceToHost);
    }
    else {
        err = cudaMemcpy(&res, d_res2, sizeof(T), cudaMemcpyDeviceToHost);
    }
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

    err = cudaFree(d_res1);
    if (err != cudaSuccess) {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaFree(d_res2);
    if (err != cudaSuccess) {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    return res;
}

template<typename T>
void one_vector_test(int vector_length, T p, T error) {
    std::vector <T> vec(vector_length);

    for (int i = 0; i < vector_length; i++) {
        vec[i] = (T)1;
    }

    auto res_gpu = gpu_lp(vec.data(), vector_length, p);
    auto res_corr = std::pow(vector_length, (T)(1.0 / p));

    if (std::abs(res_corr - res_gpu) < error) {
        std::cout << "The one-vector test with length " << vector_length << " and p = " << p << " was successful." << std::endl;
    }
    else {
        std::cout << "The random vector test with length " << vector_length << " and p = " << p << " failed." << std::endl;
        std::cout << "The GPU result: " << res_gpu << std::endl;
        std::cout << "The analytical reference: " << res_corr << std::endl;
        std::cout << "The absolute difference: " << std::abs(res_gpu - res_corr) << std::endl;
    }
}

template<typename T>
void random_vector_test(int vector_length, T p, T error, int cpu_threads) {
    std::vector <T> vec(vector_length);

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(1.0, 1.0);
    for (int i = 0; i < vector_length; i++) {
        vec[i] = (T)(dist(e2));
    }

    auto res_gpu = gpu_lp(vec.data(), vector_length, p);
    auto res_cpu = parallel_lp(vec, vector_length, p, cpu_threads);

    if (std::abs(res_cpu - res_gpu) < error) {
        std::cout << "The random vector test with length " << vector_length << " and p = " << p << " was successful." << std::endl;
    }
    else {
        std::cout << "The random vector test with length " << vector_length << " and p = " << p << " failed." << std::endl;
        std::cout << "The GPU result: " << res_gpu << std::endl;
        std::cout << "The CPU reference: " << res_cpu << std::endl;
        std::cout << "The absolute difference: " << std::abs(res_gpu - res_cpu) << std::endl;
    }

}


int main() {
    std::cout << "Validating the GPU calculations..." <<std::endl;
    
    one_vector_test(100'000, 1.0f, 1e-4f);
    one_vector_test(100'000, 2.0, 1e-6);
    
    for (int i = 1; i < 10; i++) {
        if (i % 2 == 0) {
            random_vector_test(1'000'000, (float)i, 1e-4f, 12);
        }
        else {
            random_vector_test(10'000'000, (double)i, 1e-6, 12);
        }
    }

    return 0;

}