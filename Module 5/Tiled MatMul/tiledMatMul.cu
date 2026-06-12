%%writefile tiledMatMul.cu

#include <iostream>
#include <cuda_runtime.h>


#define TILE_WIDTH 16

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}


__global__ void MatrixMulTiledKernel(float* M, float* N, float* P, int Width)
{
    __shared__ float t_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float  t_N[TILE_WIDTH][TILE_WIDTH];


    int bX = blockIdx.x;
    int bY = blockIdx.y;
    int tX = threadIdx.x;
    int tY = threadIdx.y;

    int row = bY * TILE_WIDTH + tY;
    int col = bX * TILE_WIDTH + tX;

    float Pvalue = 0;
    for (int p = 0; p < Width/TILE_WIDTH; p++)
    {
        t_M[tY][tX] = M[row * Width + TILE_WIDTH * p + tX];
        t_N[tY][tX] = N[(p*TILE_WIDTH +tY) * Width + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) 
        {
            Pvalue += t_M[tY][k] * t_N[k][tX];
        }
        __syncthreads();
    }
    P[row * Width + col] = Pvalue;

}

void printSampleMatrix(float* matrix, int width, const char* label) {
    std::cout << "\n=== " << label << " (First 4x4 block) ===" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {

    int Width = 32; 
    size_t bytes = Width * Width * sizeof(float);

    // Allocate host memory
    float* h_M = (float*)malloc(bytes);
    float* h_N = (float*)malloc(bytes);
    float* h_P_naive = (float*)malloc(bytes);
    float* h_P_tiled = (float*)malloc(bytes);

    // Initialize matrices
    for (int i = 0; i < Width * Width; i++) {
        h_M[i] = 1.0f;
        h_N[i] = 2.0f;
    }

    // Allocate device memory
    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, bytes);
    cudaMalloc(&d_N, bytes);
    cudaMalloc(&d_P, bytes);

    // Copy inputs to device
    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

    // Define execution configuration using TILE_WIDTH
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((Width + blockDim.x - 1) / blockDim.x,
                 (Width + blockDim.y - 1) / blockDim.y);

    cudaMemset(d_P, 0, bytes); // Clear device output matrix
    MatrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, Width);
    cudaMemcpy(h_P_naive, d_P, bytes, cudaMemcpyDeviceToHost);
    printSampleMatrix(h_P_naive, Width, "Testing Naive Implementation");

    cudaMemset(d_P, 0, bytes); // Clear device output matrix again
    MatrixMulTiledKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, Width);
    cudaMemcpy(h_P_tiled, d_P, bytes, cudaMemcpyDeviceToHost);
    printSampleMatrix(h_P_tiled, Width, "Testing Tiled Implementation");

    bool correct = true;
    for (int i = 0; i < Width * Width; i++) {
        if (h_P_naive[i] != h_P_tiled[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "\n>>> Verification: " << (correct ? "SUCCESS! Matrices match." : "FAIL! Outputs differ.") << " <<<" << std::endl;

    // Cleanup
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P_naive);
    free(h_P_tiled);

    return 0;
}