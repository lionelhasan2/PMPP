
#define TILE_WIDTH 16



#include <iostream>
#include <cuda_runtime.h>

// Tiled Matrix Multiplication Kernel (M is Row-Major, N is Column-Major)
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
        // FIXED: Changed () to [] and mapped correctly to column-major layout
        if (((ph * TILE_WIDTH) + ty) < n && col < o) {
            Nds[ty][tx] = N[col * n + (ph * TILE_WIDTH) + ty];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_WIDTH; k++) {
            // FIXED: Changed Nds[k][ty] to Nds[k][tx] to properly match the thread's column
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    // Write the result to global memory (P is Row-Major)
    if (row < m && col < o) {
        // FIXED: Changed PValue to Pvalue
        P[row * o + col] = Pvalue;
    }
}

// CPU Matrix Multiplication for verification (M is Row-Major, N is Column-Major)
void cpuMatMul(float* M, float* N, float* P, int m, int n, int o) {
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < o; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                // N is column-major, so index is [col * num_rows + row] -> [c * n + k]
                sum += M[r * n + k] * N[c * n + k];
            }
            P[r * o + c] = sum;
        }
    }
}

int main() {
    // Define non-square matrix dimensions to thoroughly test boundary conditions
    int m = 40; // Rows of M
    int n = 30; // Cols of M / Rows of N
    int o = 50; // Cols of N

    size_t size_M = m * n * sizeof(float);
    size_t size_N = n * o * sizeof(float);
    size_t size_P = m * o * sizeof(float);

    // Allocate Host Memory
    float* h_M = (float*)malloc(size_M);
    float* h_N = (float*)malloc(size_N);
    float* h_P_gpu = (float*)malloc(size_P);
    float* h_P_cpu = (float*)malloc(size_P);

    // Initialize matrices with random small float values
    for (int i = 0; i < m * n; ++i) h_M[i] = (float)(rand() % 10) / 2.0f;
    for (int i = 0; i < n * o; ++i) h_N[i] = (float)(rand() % 10) / 2.0f;

    // Allocate Device Memory
    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, size_M);
    cudaMalloc(&d_N, size_N);
    cudaMalloc(&d_P, size_P);

    // Copy data from Host to Device
    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);

    // Configure Execution Grid
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    std::cout << "Launching Column-Major N kernel with grid size (" << dimGrid.x << ", " << dimGrid.y 
              << ") and block size (" << dimBlock.x << ", " << dimBlock.y << ")..." << std::endl;

    // Launch Kernel
    TiledMatMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

    // Check for launch or execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to Host
    cudaMemcpy(h_P_gpu, d_P, size_P, cudaMemcpyDeviceToHost);

    // Verify results using CPU implementation
    cpuMatMul(h_M, h_N, h_P_cpu, m, n, o);

    bool success = true;
    for (int i = 0; i < m * o; ++i) {
        if (std::abs(h_P_gpu[i] - h_P_cpu[i]) > 1e-4) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_P_gpu[i] << ", CPU=" << h_P_cpu[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Success! Mixed layout (M row-major, N col-major) matches CPU baseline perfectly." << std::endl;
    }

    // Free Memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P_gpu);
    free(h_P_cpu);

    return 0;
}