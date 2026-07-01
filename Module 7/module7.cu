#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// Define constants for your filters and dimensions
#define FILTER_RADIUS 2
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)
#define TILE_WIDTH 16

// Constant memory for the masks (often used in convolution to speed up access)
__constant__ float d_Mask1D[FILTER_WIDTH];
__constant__ float d_Mask2D[FILTER_WIDTH * FILTER_WIDTH];

__global__ void convolution1D_kernel(const float* __restrict__ d_in, float* __restrict__ d_out, int width) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (g_idx < width) {
        float sum = 0.0f;
        for (int i = 0; i < FILTER_WIDTH; i++)
        {
            if ((g_idx + i - FILTER_RADIUS >= 0) and (g_idx + i - FILTER_RADIUS) < width)
            {
                sum += d_in[g_idx+i-FILTER_RADIUS] * d_Mask1D[i];
            }
        }
        d_out[g_idx] = sum;
    }
}

__global__ void convolution2D_kernel(const float* __restrict__ d_in, float* __restrict__ d_out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        // TODO: Implement basic 2D Convolution
        // Read from d_in, use d_Mask2D, and write to d_out[row * width + col]
        // Loop through the filter rows and columns
        
        d_out[row * width + col] = d_in[row * width + col]; // Placeholder
    }
}

__global__ void convolution2D_tiled_kernel(const float* __restrict__ d_in, float* __restrict__ d_out, int width, int height) {
    // Shared memory allocation accommodating the tile and the filter halo
    // Total shared memory tile width = TILE_WIDTH + 2 * FILTER_RADIUS
    __shared__ float s_Data[TILE_WIDTH + 2 * FILTER_RADIUS][TILE_WIDTH + 2 * FILTER_RADIUS];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: 1. Load data into shared memory (including the halo cells)
    // TODO: 2. Synchronize threads (__syncthreads())
    // TODO: 3. Compute convolution using shared memory
    // TODO: 4. Write output to global memory

    if (row < height && col < width) {
        d_out[row * width + col] = d_in[row * width + col]; // Placeholder
    }
}


int main() {
    // Dimensions
    const int width = 64;
    const int height = 64;
    const int size_1d = width * sizeof(float);
    const int size_2d = width * height * sizeof(float);

    // Host Arrays
    float h_in1D[width];
    float h_out1D[width];
    float h_mask1D[FILTER_WIDTH];

    float* h_in2D = (float*)malloc(size_2d);
    float* h_out2D = (float*)malloc(size_2d);
    float h_mask2D[FILTER_WIDTH * FILTER_WIDTH];

    // Initialize Host Data with dummy values
    for (int i = 0; i < width; i++) h_in1D[i] = 1.0f;
    for (int i = 0; i < FILTER_WIDTH; i++) h_mask1D[i] = 1.0f / FILTER_WIDTH;

    for (int i = 0; i < width * height; i++) h_in2D[i] = 1.0f;
    for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; i++) h_mask2D[i] = 1.0f / (FILTER_WIDTH * FILTER_WIDTH);

    // Device Pointers
    float *d_in1D, *d_out1D;
    float *d_in2D, *d_out2D;

    // Allocate Device Memory
    cudaMalloc(&d_in1D, size_1d);
    cudaMalloc(&d_out1D, size_1d);
    cudaMalloc(&d_in2D, size_2d);
    cudaMalloc(&d_out2D, size_2d);

    // Copy Data to Device
    cudaMemcpy(d_in1D, h_in1D, size_1d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2D, h_in2D, size_2d, cudaMemcpyHostToDevice);

    // Copy Masks to Constant Memory
    cudaMemcpyToSymbol(d_Mask1D, h_mask1D, FILTER_WIDTH * sizeof(float));
    cudaMemcpyToSymbol(d_Mask2D, h_mask2D, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));

    // ------------------------------------------------------------------------
    // Launch 1D Convolution
    // ------------------------------------------------------------------------
    int blockSize1D = 256;
    int gridSize1D = (width + blockSize1D - 1) / blockSize1D;
    
    convolution1D_kernel<<<gridSize1D, blockSize1D>>>(d_in1D, d_out1D, width);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out1D, d_out1D, size_1d, cudaMemcpyDeviceToHost);
    std::cout << "1D Convolution executed successfully." << std::std::std::endl;

    // ------------------------------------------------------------------------
    // Launch 2D Convolution (Basic)
    // ------------------------------------------------------------------------
    dim3 blockSize2D(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize2D((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    convolution2D_kernel<<<gridSize2D, blockSize2D>>>(d_in2D, d_out2D, width, height);
    cudaDeviceSynchronize();
    std::cout << "2D Basic Convolution executed successfully." << std::std::endl;

    // ------------------------------------------------------------------------
    // Launch 2D Convolution (Tiled)
    // ------------------------------------------------------------------------
    convolution2D_tiled_kernel<<<gridSize2D, blockSize2D>>>(d_in2D, d_out2D, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out2D, d_out2D, size_2d, cudaMemcpyDeviceToHost);
    std::cout << "2D Tiled Convolution executed successfully." << std::std::endl;

    // Clean up
    cudaFree(d_in1D);
    cudaFree(d_out1D);
    cudaFree(d_in2D);
    cudaFree(d_out2D);
    free(h_in2D);
    free(h_out2D);

    return 0;
}