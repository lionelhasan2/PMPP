# Convolution & GPU Programming Workbook

---

### Exercise 1
Calculate the `P[0]` value in Fig. 7.3.

**Answer:**
<br><br><br>


---

### Exercise 2
Consider performing a 1D convolution on array `N = {4,1,3,2,3}` with filter `F = {2,1,4}`. What is the resulting output array?

**Answer:**
<br><br><br>

---

### Exercise 3
What do you think the following 1D convolution filters are doing?

**a.** `[0 1 0]`
**Answer:**

<br>

**b.** `[0 0 1]`
**Answer:**

<br>

**c.** `[1 0 0]`
**Answer:**

<br>

**d.** `[-1/2 0 1/2]`
**Answer:**

<br>

**e.** `[1/3 1/3 1/3]`
**Answer:**

<br><br>

---

### Exercise 4
Consider performing a 1D convolution on an array of size `N` with a filter of size `M`:

**a.** How many ghost cells are there in total?
**Answer:**

<br>

**b.** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?
**Answer:**

<br>

**c.** How many multiplications are performed if ghost cells are not treated as multiplications?
**Answer:**

<br><br>

---

### Exercise 5
Consider performing a 2D convolution on a square matrix of size `N × N` with a square filter of size `M × M`:

**a.** How many ghost cells are there in total?
**Answer:**

<br>

**b.** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?
**Answer:**

<br>

**c.** How many multiplications are performed if ghost cells are not treated as multiplications?
**Answer:**

<br><br>

---

### Exercise 6
Consider performing a 2D convolution on a rectangular matrix of size `N₁ × N₂` with a rectangular mask of size `M₁ × M₂`:

**a.** How many ghost cells are there in total?
**Answer:**

<br>

**b.** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?
**Answer:**

<br>

**c.** How many multiplications are performed if ghost cells are not treated as multiplications?
**Answer:**

<br><br>

---

### Exercise 7
Consider performing a 2D tiled convolution with the kernel shown in Fig. 7.12 on an array of size `N × N` with a filter of size `M × M` using an output tile of size `T × T`:

```c
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P,
                                                    int width, int height) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    
    //loading input tile
    __shared__ N_s[IN_TILE_DIM][IN_TILE_DIM];
    if(row>=0 && row<height && col>=0 && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
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
**Answer:**

<br>

**b.** How many threads are needed per block?
**Answer:**

<br>

**c.** How much shared memory is needed per block?
**Answer:**

<br>

**d.** Repeat the same questions if you were using the kernel in Fig. 7.15.
**Answer:**

<br><br>

---

### Exercise 8
Revise the 2D kernel in Fig. 7.7 to perform 3D convolution.

**Answer:**
<br><br><br><br><br>

---

### Exercise 9
Revise the 2D kernel in Fig. 7.9 to perform 3D convolution.

**Answer:**
<br><br><br><br><br>

---

### Exercise 10
Revise the tiled 2D kernel in Fig. 7.12 to perform 3D convolution.

**Answer:**
<br><br><br><br><br>