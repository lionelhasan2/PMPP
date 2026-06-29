# Convolution Exercises

## Exercise 1

Calculate the `P[0]` value in Fig. 7.3.

> The value would be P[0] = 0 * f[0] + 0 * f[1] + x[0] * f[2] + x[1] * f[3] + x[2] * f[4] = 0 + 0 + (8*5) + (2*3) + (5*1) = 51 

---

## Exercise 2

Consider performing a 1D convolution on array `N = {4, 1, 3, 2, 3}` with filter `F = {2, 1, 4}`. What is the resulting output array?

> P[0] = (0 * 2) + (4 * 1) + (4 * 1) = 8
> P[1] = (4*2) + (1 * 1) + (3 * 4) = 21
> P[2] = (1 * 2) + (3 * 1) + (2 * 4) = 13
> P[3] = (3 * 2) + (2 * 1) + (3 * 4) = 20
> P[4] = (2 * 2) + (3 * 1) + (0 * 4) = 7
>
> Therefore, the resulting array would be [8,21,13,20,7]

---

## Exercise 3

What do you think the following 1D convolution filters are doing?

**a.** `[0 1 0]`
> Not modifying the input at all, no weight is given to any element other than the one at the selected indice.

**b.** `[0 0 1]`
> Shifting the values to the left because the value to the right of the selected indice gets taken.

**c.** `[1 0 0]`
> Shifting the values to the right because the value to the left of the selected indice gets taken.

**d.** `[-1/2 0 1/2]`
> ? 

**e.** `[1/3 1/3 1/3]`
> Takes the average of the surrounding values including it's own for each selected indice.

---

## Exercise 4

Consider performing a 1D convolution on an array of size `N` with a filter of size `M`:

**a.** How many ghost cells are there in total?
> The number of ghost cells at each end would be the number of radius cells.
> So, 2 * ((M - 1) / 2) = M - 1

**b.** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?
> Each cell completes M multiplications, so (N  * M). 

**c.** How many multiplications are performed if ghost cells are not treated as multiplications?
> The number of multiplications performed gets tricky as the first element looses r multiplications while the second element loses r - 1 multiplications and so on until the rth element which loses 1 multiplication. The same happens at the end of the array. So, the total number of multiplications performed would be (N * M) - (2 * (r * (r + 1) / 2)) = (N * M) - r(r + 1)

---

## Exercise 5

Consider performing a 2D convolution on a square matrix of size `N × N` with a square filter of size `M × M`:

**a.** How many ghost cells are there in total?
>  The number of ghost cells would be the number of radius cells at each of the four directions 4 * N * r, combined with the fact that there are r * r ghost cells at each corner. Therefore there is (4 * N * r) + ( 4* r * r) = (4 * r) (N + r) 
>   
**b.** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?
> Each cell completes M*M multiplications therefore the number of multiplications performed is N * N * M * M. 

**c.** How many multiplications are performed if ghost cells are not treated as multiplications?
> 

---


## Exercise 7

Consider performing a 2D tiled convolution with the kernel shown in Fig. 7.12 on an array of size `N × N` with a filter of size `M × M` using an output tile of size `T × T`:

```cuda
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P,
                                                    int width, int height) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    
    //loading input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if(row>=0 && row<height && col>=0 && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    
    // turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >=0 && row < height) {
        if (tileCol>=0 && tileCol<OUT_TILE_DIM && tileRow>=0
                && tileRow<OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    Pvalue += F_c[fRow][fCol]*N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*width+col] = Pvalue;
        }
    }
}
```

**a.** How many thread blocks are needed?
> The size of each thread block is dictated by the dimensions of the input tile, but each thread block can only calculate the number of elements inside the output tile because input tiles contain halo cells meaning certain threads need to be turned off during the calculation stage. 
>
> Total Blocks Needed = (N * N) / ( T * T)
>
**b.** How many threads are needed per block?
> The number of threads per block is again dictated by the dimensions of the input tile, in this case the dimensions of the input tile is ((T+2r) * (T*2r))where r is the radius of the filter or ((M-1) / 2). 

**c.** How much shared memory is needed per block?
> Shared memory must hold the entire input, so (T+2r) * (T + 2r) * 4B.


**d.** Repeat the same questions if you were using the kernel in Fig. 7.15.
> A) The number of output elements computed by a thread block is still dictated by the size of the output tile. Therefore, Total Blocks Needed = ( N * N) / (T *T)

> B) The number of threads needed per block reduces to only (T * T) because the size is dictated by the size of the output tile 

> C) Each input tile loads in a single element into shared memory while the halo cells have their values retrieved from constant caches, therefore the shared memory needed = T * T * 4B. 
---

## Exercise 8

Revise the 2D kernel in Fig. 7.7 to perform 3D convolution.

> 

---

## Exercise 9

Revise the 2D kernel in Fig. 7.9 to perform 3D convolution.

> 

---

## Exercise 10

Revise the tiled 2D kernel in Fig. 7.12 to perform 3D convolution.

>