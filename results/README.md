# Hardware used for testing

I ran all of the tests on my PC, which has an AMD Ryzen 5 3600X CPU (6 cores, 12 threads) and an NVIDIA GeForce RTX 2070 Super GPU (8GB VRAM).

I the CPU version of my prgoram with 1,2,...,12 threads and you can see the results on the following images:

# thread_vs_t_cpu_mean.png
The average time required to compute the 2-norm of an 1,000,000 long vector, from 100 runs (the first 10 are discarded).

# thread_vs_t_cpu_min.png
The minimum time required to compute the 2-norm of an 1,000,000 long vector, from 100 runs.

# p_vs_t_cpu_mean.png
The average time required to compute the p-norms (for p=1,2,3,4,5) of an 1,000,000 long vector, from 100 runs, with 1,6,12 threads (the first 10 are discarded).

# p_vs_t_cpu_min.png
The minimum time required to compute the p-norms (for p=1,2,3,4,5) of an 1,000,000 long vector, from 100 runs, with 1,6,12 threads.

And I also compared the CPU times with the GPU times. You can see the result on the following image:

# cpu_vs_gpu_total_mean.png
The average time required to compute the p-norms (for p=1,2,3,4,5) of a 10,000,000 long vector, from 100 runs with 12 threads on the CPU and 512 threads on GPU.

# cpu_vs_gpu_total_min.png
The minimum time required to compute the p-norms (for p=1,2,3,4,5) of a 10,000,000 long vector, from 100 runs with 12 threads on the CPU and 512 threads on GPU.

# cpu_vs_gpu_calconly_mean.png
The average time required to compute the p-norms (for p=1,2,3,4,5) of a 10,000,000 long vector, from 100 runs with 12 threads on the CPU and a thread-size of 512 on GPU. The time of the memory allocation, copying and memory freeing is not included.

# cpu_vs_gpu_calconly_min.png
The minimum time required to compute the p-norms (for p=1,2,3,4,5) of a 10,000,000 long vector, from 100 runs with 12 threads on the CPU and a thread-size of 512 on GPU. The time of the memory allocation, copying and memory freeing is not included.

# Conclusion
- I expected the time to be monotone decreasing, but there is a "bump" at 6 threads (which is the number of my CPU cores).
- I expected the float calculations being faster, but they  were constantly slower. I think it can be caused by a double --> float conversion, but could ot find the source.
- The float and double calculations have similar speed on the CPU, while on the GPU, the float is much faster.