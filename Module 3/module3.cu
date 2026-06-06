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
    for (int row = 0; row < Width; ++row) {
        if ((col < Width)){
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row*Width+k]*N[k*Width+col];
            }
            P[row*Width+col] = Pvalue;
        }
    }
}


int main() {
    int Width = 32;
    size_t bytes = Width * Width * sizeof(float);

    // Allocate host memory
    float* h_M = (float*) malloc(bytes);
    float* h_N = (float*) malloc(bytes);
    float* h_P = (float*) malloc(bytes);
    // Initialize matrices

    for (int i = 0; i < Width * Width; i ++)
    {
        h_M[i] = 1.0;
        h_N[i] = 2.0;
    }
    // Allocate device memory
    float  *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, bytes);
    cudaMalloc(&d_N, bytes);
    cudaMalloc(&d_P, bytes);

    // Copy to device
    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

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

    // Cleanup
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}