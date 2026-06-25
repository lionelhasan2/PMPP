# CUDA Memory & Tiling Exercises

## Exercise 1

Write a matrix multiplication kernel function that corresponds to the design illustrated in Fig. 6.4.

```cuda
__global__ void TiledMatMulCornerTurning(
    float *A,   // row-major,    shape [m x n]
    float *B,   // col-major,    shape [n x o]  → stored as B[col * n + row]
    float *C,   // row-major,    shape [m x o]
    int m, int n, int o)
{
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y,  bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;   // output row in C
    int col = bx * TILE_WIDTH + tx;   // output col in C

    float sum = 0.0f;

    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++)
    {
        int aCol = ph * TILE_WIDTH + tx;
        As[ty][tx] = (row < m && aCol < n) ? A[row * n + aCol] : 0.0f;

        // B is column-major: corner turning 

    
        int bRow = ph  * TILE_WIDTH + tx;   // row index into B  (driven by tx → coalesced)
        int bCol = bx  * TILE_WIDTH + ty;   // col index into B  (driven by ty → same col group)
        Bs[tx][ty] = (bRow < n && bCol < o) ? B[bCol * n + bRow] : 0.0f;
        //  ^^^ transposed store: element that belongs at logical [k][tx]
        //      is written to Bs[tx][ty] and read back as Bs[k][tx] below
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < m && col < o)
        C[row * o + col] = sum;
}


```

---

## Exercise 2

For tiled matrix multiplication, of the possible range of values for `BLOCK_SIZE`, for what values of `BLOCK_SIZE` will the kernel completely avoid uncoalesced accesses to global memory? *(You need to consider only square blocks.)*

Considering `BLOCK_SIZE` represents the width of the thread block and warps consist of 32 threads in cuda, BLOCK_SIZE must be a multiple of 32 to completely avoid uncoalesced accesses to global memory. This allows each thread in each warp to access adjacent memory spaces by accessing values all in the same row of the matrix. Values less than 32 would cause warps to access multiple rows of the matrix preventing all adjacent threads from accessing adjacent memory locations. A CUDA thread block is limited to 1024 threads only, so the only possible value allowed for this question is 32 because 32 x 32 = 1024. 

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

**a.** The access to array `a` of line 06:
> Coalesced because neighbouring threads have will access neighbouring memory locations as they will have subsequent `threadIdx.x` values. 

**b.** The access to array `a_s` of line 06
> Not applicable to accessing shared memory. 

**c.** The access to array `b` of line 08:
> Coalesced, for neighbouring threads within a block only the `threadIdx.x` value will change.

**d.** The access to array `c` of line 08:
>  Not coalesced, neighbouring threads will have a stride of 4 between their memory accesses. 

**e.** The access to array `bc_s` of line 08
> Not applicable to accessing shared memory.

**f.** The access to array `a_s` of line 12
> Not applicable to accessing shared memory.

**g.** The access to array `d` of line 12: 
> Coalesced, neighbouring threads will access neighbouring memory locations. 

**h.** The access to array `bc_s` of line 13: 
>  Not applicable to accessing shared memory. 

**i.** The access to array `e` of line 13: 
>  Not coalesced, the stride between memory locations accessed between neighbouring threads in a block will be 8. 

---

## Exercise 4

What is the floating point to global memory access ratio (in OP/B) of each of the following matrix-matrix multiplication kernels? 

*Assume we have a matrix `M` of size (m, n) and a matrix `N` of size (n, o). We assume `float32` (4 bytes) as the data type.*

**a.** The simple kernel described in Chapter 3, Multidimensional Grids and Data, without any optimizations applied.
> The number of floating point operations that occur in each thread is:
>  n multplications between each element in M's row and N's col
>  n additions summing the result of the multiplication with the current total. 
>
> Therefore the total amount of floating point operations is 2n. 
> 
> The number of memory accesses that occur is:
> n for the row that is loaded from M
> n for the row that is loaded from N 
>
> Therefore the total amount of memory accesses is 2n, and the amount of bytes accessed is 4 B * 2n = 8B * n
>
> The global memory access ratio in this scenario is : number of floating point operations / number of bytes accessed = (2 Ops * n) / (8 B * n) = 0.25  Ops/B 


**b.** The kernel described in Chapter 5, Memory Architecture and Data Locality, with shared memory tiling applied using a tile size of 32 × 32.
> The number of floating operations that occurs remains the same at 2 * n. 
>
> The number of memory accesses is divided by the tile width of 32 because each thread has access to 32 elements from global memory for each element that it accesses. Therefore number of memory accesses is 2n/32 = n/16 
>
> The global memory access ratio is (2 Ops * n) / ( (n * 4B) / 16) = 8 Ops/B 




**c.** The kernel described in this chapter with shared memory tiling applied using a tile size of 32 × 32 and thread coarsening applied using a coarsening factor of 4.        
> The number of floating operations that occurs remains the same at 2 * n. 
>
> The number of memory accesses for the M matrix reduces by 4 times more than the number of memory accesses for the tiled implementation because each thread reuses the memory accesses for 4 outputs rather than only 1. Therefore, the number of memory accesses for M is (n / (32*4)) = n / 128 and the number of memory accesses for N remains at n / 32. The total number of memory accesses is (n/32) + (n/128) = (5n/128)
>
> The global memory access ratio is (2 Ops * n) / ((5n * 4B) / 128) = 12.8 Ops/B. 
>
> 