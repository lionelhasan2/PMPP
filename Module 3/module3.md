# Module 3

## Exercise 1

### a. One thread per matrix row

Each thread computes one output row, so the execution configuration should be one-dimensional. A simple launch uses a 1D block and grid sized to cover the number of rows.

```cuda
__global__
void matrixMulRowKernel(float* M, float* N, float* P, int size) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < size) {
		for (int col = 0; col < size; ++col) {
			float sum = 0;
			for (int j = 0; j < size; ++j) {
				sum += M[row * size + j] * N[j * size + col];
			}
			P[row * size + col] = sum;
		}
	}
}

dim3 blockDimRow(256);
dim3 gridDimRow((size + blockDimRow.x - 1) / blockDimRow.x);
```

### b. One thread per matrix column

This version also uses a 1D launch. Each thread computes one output column, which is a symmetric design to the row-based kernel.

```cuda
__global__
void matrixMulColKernel(float* M, float* N, float* P, int size) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < size) {
		for (int row = 0; row < size; ++row) {
			float sum = 0;
			for (int j = 0; j < size; ++j) {
				sum += M[row * size + j] * N[j * size + col];
			}
			P[row * size + col] = sum;
		}
	}
}

dim3 blockDimCol(256);
dim3 gridDimCol((size + blockDimCol.x - 1) / blockDimCol.x);
```

### c. Pros and cons

For non-square matrices, the kernel that launches more threads will likely be better than the kernel that launches less as it allows for more work to be completed in parallel. If the matrix is wider than it is taller then the kernel that launches one thread per column will be more advantageous and vice-versa if the opposite is true. 

In terms of memory accesses, the column-major implementation is superior because threads will compute adjacent elements in memory due to a row-major layout at the same time. The hardware may then be able to combine these memory accesses into a single efficient one. 


## Exercise 2

Matrix-vector multiplication assigns one thread to each output element. The kernel below follows that design and is the version I used in the notebook.

```cuda
__global__ void matrixVecMulKernel(float* B, float* c, float* result, int vector_size, int matrix_rows) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < matrix_rows) {
		float sum = 0;
		for (int j = 0; j < vector_size; ++j) {
			sum += B[i * vector_size + j] * c[j];
		}
		result[i] = sum;
	}
}

dim3 blockDimVec(256);
dim3 gridDimVec((matrix_rows + blockDimVec.x - 1) / blockDimVec.x);
```

## Exercise 3

Consider the following CUDA kernel and the corresponding host function that calls it:

```cuda
01 __global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
02     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
03     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
04     if (row < M && col < N) {
05         b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
06     }
07 }
08 void foo(float* a_d, float* b_d) {
09     unsigned int M = 150;
10     unsigned int N = 300;
11     dim3 bd(16, 32);
12     dim3 gd((N - 1) / 16 + 1, ((M - 1) / 32 + 1));
13     foo_kernel <<< gd, bd >>> (a_d, b_d, M, N);
14 }
```

### 3.1

**a.** What is the number of threads per block?

There are 16 threads in the x-dimension and 32 threads in the y-dimension, so there are 16 x 32 = 512 threads per block.

**b.** What is the number of threads in the grid?

There are 19 blocks in the x-dimension and 5 blocks in the y-dimension, so there are 19 x 5 x 512 = 48,640 threads in the grid.

**c.** What is the number of blocks in the grid?

There are 19 x 5 = 95 blocks in the grid.

**d.** What is the number of threads that execute the code on line 05?

Only threads that satisfy row < M and col < N execute line 05. Since M = 150 and N = 300, the number of threads that actually execute that line is 150 x 300 = 45,000.

## Exercise 4

Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10.

**a.** If the matrix is stored in row-major order.

Row-major indexing uses row x width + col, so the index is 20 x 400 + 10 = 8,010.

**b.** If the matrix is stored in column-major order.

Column-major indexing uses col x height + row, so the index is 10 x 500 + 20 = 5,020.

## Exercise 5

Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5.

For row-major storage, the index is z x (width x height) + y x width + x. Substituting the values gives 5 x (400 x 500) + 20 x 400 + 10 = 1,008,010.

