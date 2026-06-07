%%writefile MatrixMulKernel.cu

#include <iostream>

#include <iostream>
#include <cuda_runtime.h>

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row*Width+k]*N[k*Width+col];
        }
        P[row*Width+col] = Pvalue;
    }
}

//Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.

__global__ void MatrixMulKernelRow(float* M, float* N, float* P, int Width)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    for (int col = 0; col < Width; ++col) {
        if ((row < Width)){
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row*Width+k]*N[k*Width+col];
            }
            P[row*Width+col] = Pvalue;
        }
    }
}

// Write a kernel that has each thread produce one output matrix column. 
// Fill in the execution configuration parameters for the design.
__global__ void MatrixMulKernelCol(float* M, float* N, float* P, int Width)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if ((col < Width))
    { 
        for (int row = 0; row < Width; ++row) 
        {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) 
            {
                Pvalue += M[row*Width+k]*N[k*Width+col];
            }
            P[row*Width+col] = Pvalue;
        }
    }
}

// Matrix Vector multiplication kernel. Each thread computes one element of the output vector.

__global__ void MatrixVectorMulKernel(float* B, float* C, float* A, int width)
{
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    if (row < width)
    {
        float Pvalue = 0;
        for (int i = 0; i < width; ++i)
        {
            Pvalue+= B[row*width+i] * C[i];
        }
        A[row] = Pvalue;
    }
}


int main() {
    int Width = 32;
    size_t bytes = Width * Width * sizeof(float);

    // Allocate host memory
    float* h_M = (float*) malloc(bytes);
    float* h_N = (float*) malloc(bytes);
    float* h_P = (float*) malloc(bytes);
    float* h_C = (float*) malloc(Width * sizeof(float));
    float* h_A = (float*) malloc(Width * sizeof(float));

    float* h_B = (float*) malloc(bytes);
    // Initialize matrices

    for (int i = 0; i < Width * Width; i ++)
    {
        h_M[i] = 1.0;
        h_N[i] = 2.0;
        h_B[i] = 1.0;
    }

    // Initialize Vector
    for (int i = 0; i < Width; i++)
    {
        h_C[i] = 2.0;
    }

    // Allocate device memory
    float  *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, bytes);
    cudaMalloc(&d_N, bytes);
    cudaMalloc(&d_P, bytes);

    float *d_B, *d_C, *d_A;
    cudaMalloc(&d_B,bytes);
    cudaMalloc(&d_C, Width * sizeof(float));
    cudaMalloc(&d_A, Width * sizeof(float));


    // Copy to device
    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, Width * sizeof(float), cudaMemcpyHostToDevice);


    dim3 blockDim(16,16);
    dim3 gridDim((Width + blockDim.x - 1) / blockDim.x,
                 (Width + blockDim.y - 1) / blockDim.y);

    // Test 1: MatrixMulKernel (2D element-wise)
    std::cout << "\n=== Testing MatrixMulKernel (2D element-wise) ===" << std::endl;
    MatrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, Width);
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
    std::cout << "Result (first 4x4 block, each should be 64.0):" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << h_P[i*Width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Test 2: MatrixMulKernelRow (each thread produces one row)
    std::cout << "\n=== Testing MatrixMulKernelRow (each thread produces one row) ===" << std::endl;
    dim3 blockDim_row(256);
    dim3 gridDim_row((Width + blockDim_row.x - 1) / blockDim_row.x);
    MatrixMulKernelRow<<<gridDim_row, blockDim_row>>>(d_M, d_N, d_P, Width);
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
    std::cout << "Result (first 4x4 block, each should be 64.0):" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << h_P[i*Width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Test 3: MatrixMulKernelCol (each thread produces one column)
    std::cout << "\n=== Testing MatrixMulKernelCol (each thread produces one column) ===" << std::endl;
    dim3 blockDim_col(256);
    dim3 gridDim_col((Width + blockDim_col.x - 1) / blockDim_col.x);
    MatrixMulKernelCol<<<gridDim_col, blockDim_col>>>(d_M, d_N, d_P, Width);
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
    std::cout << "Result (first 4x4 block, each should be 64.0):" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << h_P[i*Width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Test 4: MatrixVectorMul
    std::cout << "\n=== Testing MatrixVectorMul (Matrix-Vector Multiplication) ===" << std::endl;
    dim3 blockDimVec(256);
    dim3 gridDimVec((Width + blockDimVec.x - 1) / blockDimVec.x);
    MatrixVectorMulKernel<<<gridDimVec, blockDimVec>>>(d_B, d_C, d_A, Width);
    cudaMemcpy(h_A, d_A, Width * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Result (vector A, each should be 64.0):" << std::endl;
    for (int i = 0; i < Width; i++) {
        std::cout << h_A[i] << " ";
        if ((i + 1) % 8 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;


    // Cleanup
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A);
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_B);
    free(h_C);
    free(h_A);

    return 0;
}