# Hardware used for testing

I ran all of the tests on my PC, which has a Rytwn 5 3600X CPU (6 cores, 12 threads). I tested my prgoram with 1,2,...,12 threads and you cen see the results on the following images:

# thread_vs_t.png
The average time required to compute the 2-norm of an 1000000 long vector, from 100 runs.

# p_vs_t.png
The average time required to compute the p-norms (for p=1,2,3,4,5) of an 1000000 long vector, from 100 runs, with 1,6,12 threads.

# Conclusion
- I expected the time to be monotone decreasing, but there is a "bump" at 6 threads (which is the number of my CPU cores).
- I expected the float calculations being faster, but they  were constantly slower. I think it can be caused by a double --> float conversion, but could ot find the source.