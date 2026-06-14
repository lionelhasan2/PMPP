# CUDA Memory & Tiling Exercises

## Exercise 1

Write a matrix multiplication kernel function that corresponds to the design illustrated in Fig. 6.4.

```cuda
__global__ void TiledMatMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    // Shared memory tiles
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output indices
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // Loop over tiles
    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        
        // Load M (Row-Major): Coalesced
        if (row < m && (ph * TILE_WIDTH + tx) < n) {
            Mds[ty][tx] = M[row * n + (ph * TILE_WIDTH + tx)];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        // Load N (Column-Major): Coalesced
        if (((ph * TILE_WIDTH) + ty) < n && col < o) {
            Nds[ty][tx] = N[col * n + (ph * TILE_WIDTH) + ty];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    // Write the result to global memory (P is Row-Major)
    if (row < m && col < o) {
        P[row * o + col] = Pvalue;
    }
}

```

---

## Exercise 2

For tiled matrix multiplication, of the possible range of values for `BLOCK_SIZE`, for what values of `BLOCK_SIZE` will the kernel completely avoid uncoalesced accesses to global memory? *(You need to consider only square blocks.)*




---

## Exercise 3

Consider the following CUDA kernel:

```cuda
    __global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ float a_s[256];
        __shared__ float bc_s[4 * 256];
        
        a_s[threadIdx.x] = a[i];
        for(unsigned int j = 0; j < 4; ++j) {
            bc_s[j * 256 + threadIdx.x] = b[j * blockDim.x * gridDim.x + i] + c[i * 4 + j];
        }
        __syncthreads();
        
        d[i + 8] = a_s[threadIdx.x];
        e[i * 8] = bc_s[threadIdx.x * 4];
    }
```

For each of the following memory accesses, specify whether they are **coalesced**, **uncoalesced**, or if **coalescing is not applicable**:

**a.** The access to array `a` of line 06: `a[blockIdx.x*blockDim.x + threadIdx.x]`
> 

**b.** The access to array `a_s` of line 06
> 

**c.** The access to array `b` of line 08: `b[j*blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x]`
> 

**d.** The access to array `c` of line 08: `c[(blockIdx.x*blockDim.x + threadIdx.x)*4 + j]`
> 

**e.** The access to array `bc_s` of line 08
> 

**f.** The access to array `a_s` of line 12
> 

**g.** The access to array `d` of line 12: `d[blockIdx.x*blockDim.x + threadIdx.x + 8]`
> 

**h.** The access to array `bc_s` of line 13: `bc_s[threadIdx.x*4]`
> 

**i.** The access to array `e` of line 13: `e[(blockIdx.x*blockDim.x + threadIdx.x)*8]`
> 

---

## Exercise 4

What is the floating point to global memory access ratio (in OP/B) of each of the following matrix-matrix multiplication kernels? 

*Assume we have a matrix `M` of size (m, n) and a matrix `N` of size (n, o). We assume `float32` (4 bytes) as the data type.*

**a.** The simple kernel described in Chapter 3, Multidimensional Grids and Data, without any optimizations applied.




**b.** The kernel described in Chapter 5, Memory Architecture and Data Locality, with shared memory tiling applied using a tile size of 32 × 32.




**c.** The kernel described in this chapter with shared memory tiling applied using a tile size of 32 × 32 and thread coarsening applied using a coarsening factor of 4.

