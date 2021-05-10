# The project
This is my first C++ program. It's an lp vector norm calculator. It can only utilize the CPU at the moment, but it can work in serial and parallel as well. In the future, I'm planning to add support for GPUs as well.

This is my project for my elective subject "Scientific Programming of Graphical Processors" at Eötvös Loránd University.

# My results
My project had 2 separate stages: a CPU-only part (code: /cpu/, result times: /results/data_cpu/) and a GPU part (using CUDA), with CPU comparision (code: /cuda/, result times: /results/data_gpu/).

## Hardware used for testing
- CPU: AMD Ryzen 5 3600X (6 cores, 12 threads)
- GPU: Nvidia GeForce RTX 2070 Super

## CPU
For the parallel processing, I used the async and future.

For the images, I calculated the p-norm of 1,000,000 long random vectors, 100 times for each configuration. I noticed that the first few runs were slower, so I discarded the first 10 and take the mean of the remaining 90.

Based on my runs, I was surprised finding out that using 6 threads were slightly slower than expected (as it was slower than 5 and 7 threads). It was also surprising to me to see that the double calculations were faster.

In the following figure you can see the effect of the number of threads on the running times (using the 2-norm):
![thread_vs_t_cpu](/results/thread_vs_t_cpu.png)

In the following figure you can see the times with different norms and number of threads:
![thread_vs_t_cpu](/results/p_vs_t_cpu.png)

## GPU
While doing this part, I've learned a lot from [CoffeeBeforeArch's youtube channel](https://www.youtube.com/channel/UCsi5-meDM5Q5NE93n_Ya7GA) as well. I decided to work using CUDA 

For the images, I calculated the p-norm of 100,000 long random vectors, as I've got different results with the CPU and GPU calculations for 1,000,000 long vectors (for details, see the Current problem(s) section). The first few runs were slower this time as well, so I ran everything 100 times and discarded the first 10 runs from the averages. I also used a block-size of 512 for the GPU runs, and 12 threads for the CPU runs.

In the following figure you can see how the float and double times compares to each other on CPU and GPU, with different p values:
![cpu_vs_gpu](/results/cpu_vs_gpu.png)

# Current problem(s)
- The CPU is faster in Release mode, while it works as expected in Debug mode
- The GPU and CPU gives different result for long vectors (1M for Debug, 100k for Release)

# Possible TODOs
- Run the code on a different hardware
- Make the sum-reduction even more optimized
- Move the CUDA functions into a separate file for clarity (I've heard that it's possible, but it was complicated)
- Create an OpenGL version