# Hardware used for testing

I ran all of the tests on my PC, which has an AMD Ryzen 5 3600X CPU (6 cores, 12 threads) and an NVIDIA GeForce RTX 2070 Super GPU.

I the CPU version of my prgoram with 1,2,...,12 threads and you cen see the results on the following images:

# thread_vs_t_cpu.png
The average time required to compute the 2-norm of an 1,000,000 long vector, from 100 runs.

# p_vs_t_cpu.png
The average time required to compute the p-norms (for p=1,2,3,4,5) of an 1,000,000 long vector, from 100 runs, with 1,6,12 threads.

And I also compared the CPU times with the GPU times. You can see the result on the following image:

# cpu_vs_gpu.png
The average time required to compute the p-norms (for p=1,2,3,4,5) of an 100,000 long vector, from 100 runs with 12 threads on the CPU and 512 threads on GPU (with block-size 196 and 512 threads).

# Conclusion
- I expected the time to be monotone decreasing, but there is a "bump" at 6 threads (which is the number of my CPU cores).
- I expected the float calculations being faster, but they  were constantly slower. I think it can be caused by a double --> float conversion, but could ot find the source.
- The floats are much faster on the GPU, while the doubles are much faster on the CPU.