## Exercise 1

Consider the following CUDA kernel and the corresponding host function that calls it:

```cuda
__global__ void foo_kernel(int* a, int* b) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(threadIdx.x < 40 || threadIdx.x >= 104) {
		b[i] = a[i] + 1;
	}
	if(i%2 == 0) {
		a[i] = b[i]*2;
	}
	for(unsigned int j = 0; j < 5 - (i%3); ++j) {
		b[i] += j;
	}
}

void foo(int* a_d, int* b_d) {
	unsigned int N = 1024;
	foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
}
```

**a. What is the number of warps per block?**

There are 128 threads per block and 32 threads per warp, so there are 128/32 = 4 warps per block.

**b. What is the number of warps in the grid?**

There are 8 thread blocks with 4 warps per block, therefore there is 32 total warps within the grid.

**c. For the statement on line 04:**

**i. How many warps in the grid are active?**

There are three warps that have their thread indices span over the range 0 - 40 and 104 - 128. Since there are 8 blocks, 24 warps within the grid are active.

**ii. How many warps in the grid are divergent?**

Two warps within each block are divergent as the entirety of their warp does not fit within the range 0 - 40 and 104 - 128 , the second warp with thread indices 32-63 and the fourth warp with indices 96-128 have divergent behaviour on line 04. All of warp 1 completes line 04 while all of warp 2 skips line 04. Because the grid has 8 blocks, 16 warps within the grid are divergent. 


**iii. What is the SIMD efficiency (in %) of warp 0 of block 0?**

100% because there is no control divergence within warp 0. 

**iv. What is the SIMD efficiency (in %) of warp 1 of block 0?**

8/32  = 25% because only the  first 8 threads in the warp complete line 04. 


**v. What is the SIMD efficiency (in %) of warp 3 of block 0?**

24/32 = 75% because only the first 8 threads in the warp do not complete line 04 while the other 24 do. 

**d. For the statement on line 07:**

**i. How many warps in the grid are active?**

All 32 warps within the grid are active as the condition causes every second thread to reach line 07.

**ii. How many warps in the grid are divergent?**

All 32 warps are divergent. 

**iii. What is the SIMD efficiency (in %) of warp 0 of block 0?**

There is a 50% efficiency because only half of the threads in the warp will complete the instruction.

**e. For the loop on line 09:**

**i. How many iterations have no divergence?**

The first three iterations of the loop will be completed by each thread.

**ii. How many iterations have divergence**

The last two iterations of the loop may not be completed by each thread, therefore there may be divergence. 

## Exercise 2

For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

## Exercise 3

For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?

## Exercise 4

Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time in microseconds to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads' total execution time is spent waiting for the barrier?

## Exercise 5

A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the `__syncthreads()` instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.

## Exercise 6

If a CUDA device's SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM?

a. 128 threads per block

b. 256 threads per block

c. 512 threads per block

d. 1024 threads per block

## Exercise 7

Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level.

a. 8 blocks with 128 threads each

b. 16 blocks with 64 threads each

c. 32 blocks with 32 threads each

d. 64 blocks with 32 threads each

e. 32 blocks with 64 threads each

## Exercise 8

Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

a. The kernel uses 128 threads per block and 30 registers per thread.

b. The kernel uses 32 threads per block and 29 registers per thread.

c. The kernel uses 256 threads per block and 34 registers per thread.

## Exercise 9

A student mentions that they were able to multiply two 1024 x 1024 matrices using a matrix multiplication kernel with 32 x 32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?
