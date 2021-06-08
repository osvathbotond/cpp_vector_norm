# The project
This is my first C++ program. It's an lp vector norm calculator. It can utilize both the CPU in serial and parallel and the GPU using CUDA.

This is my project for my elective subject "Scientific Programming of Graphical Processors" at Eötvös Loránd University.

# My results
My project had 2 separate stages: a CPU-only part (code: /cpu/, result times: /results/data_cpu/) and a GPU part (using CUDA), with CPU comparision (code: /cuda/, result times: /results/data_gpu/).

## Hardware used for testing
- CPU: AMD Ryzen 5 3600X (6 cores, 12 threads)
- GPU: NVIDIA GeForce RTX 2070 Super (8GB VRAM)

## CPU
For the parallel processing, I used the async and future.

For the images, I calculated the p-norm of 1,000,000 long random vectors, 100 times for each configuration. I noticed that the first few runs were slower, so I discarded the first 10 and take the mean of the remaining 90.

Based on my runs, I was surprised finding out that using 6 threads were slightly slower than expected (as it was slower than 5 and 7 threads). It was also surprising to me to see that the double calculations were faster.

In the following figures you can see the effect of the number of threads on the running times (using the 2-norm).

The mean time:
![thread_vs_t_cpu_mean](/results/thread_vs_t_cpu_mean.png)

The min time:
![thread_vs_t_cpu_min](/results/thread_vs_t_cpu_min.png)

In the following figures you can see the times with different norms and number of threads.

The mean time:
![thread_vs_t_cpu_mean](/results/p_vs_t_cpu_mean.png)

The min time:
![thread_vs_t_cpu_min](/results/p_vs_t_cpu_min.png)

## GPU
While doing this part, I've learned a lot from [CoffeeBeforeArch's youtube channel](https://www.youtube.com/channel/UCsi5-meDM5Q5NE93n_Ya7GA) as well. I decided to work using CUDA, because I have an NVIDIA GPU.

For the images, I calculated the p-norm of 10,000,000 long random vectors. I ran everything 100 times and I used a block-size of 512 for the GPU runs, and 12 threads for the CPU runs.

In the following figures you can see how the float and double times compares to each other on CPU and GPU, with different p values.

The mean time:
![cpu_vs_gpu_total_mean](/results/cpu_vs_gpu_total_mean.png)

The min time:
![cpu_vs_gpu_total_mean](/results/cpu_vs_gpu_total_min.png)

And in the following figures you can see the total times replaced with the calculation time only (i.e. the memory allocation, memory freeing and copying is not counted).

The mean time:
![cpu_vs_gpu_calconly_mean](/results/cpu_vs_gpu_calconly_mean.png)

The min time:
![cpu_vs_gpu_calconly_min](/results/cpu_vs_gpu_calconly_min.png)


# Possible TODOs
- Run the code on a different hardware
- Make the sum-reduction even more optimized
- Move the CUDA functions into a separate file for clarity (I've heard that it's possible, but it was complicated)
- Create an OpenGL version