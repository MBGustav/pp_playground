#include <cassert>
// #include <cuda_runtime.h>

#include "cudaMatrix.cuh"


namespace gpuLinalg{
    __global__ void MatrixMultiplicationOnGPU(data_t *A, data_t *B, data_t *C, 
    int m, int k, int n, 
    int ldA, int ldB, int ldC,
    data_t alpha, data_t beta)
    {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        int gx = tx + blockDim.x * blockIdx.x;
        int gy = ty + blockDim.y * blockIdx.y;
        
        __shared__ data_t SharedMemoA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ data_t SharedMemoB[BLOCK_SIZE][BLOCK_SIZE];
        
        if(gx >= m || gy >= n) return;
        
        for (int stride_x = gx; stride_x < m; stride_x += BLOCK_SIZE)
        {
            data_t acc = 0.0f;
            // for (int stride_y = gy; stride_y < n; stride_y += BLOCK_SIZE)
            {
                SharedMemoA[tx][ty] = A[idx_matrix(ldA, gy, gx)];
                SharedMemoB[tx][ty] = B[idx_matrix(ldB, gy, gx)];
                
                for (int k = 0; k < BLOCK_SIZE; ++k)
                acc += SharedMemoA[tx][k] * SharedMemoB[k][ty];
                __syncthreads();
                C[idx_matrix(ldC, gy, gx)] += alpha * acc; //beta * C[idx_matrix(ldC, gx, gy)];
            }
        }
    }
        
    // Small kernel for matricial multiplication
    __global__ void MatrixMultiplicationOnGPU_small(data_t *A, data_t *B, data_t *C,
        int m, int k, int n, int ldA, int ldB, int ldC,
        data_t alpha, data_t beta){
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int gx = tx + blockDim.x * blockIdx.x;
            const int gy = ty + blockDim.y * blockIdx.y;
            
            if(gx >= m || gy >= n) return;
            
            data_t acc = 0.0f;
            for(int kk =0; kk < k ; kk++){
                acc += A[idx_matrix(ldA, tx, kk)] * B[idx_matrix(ldB, kk, ty)];
            }
            
            C[idx_matrix(ldC, tx,ty)] = alpha * acc + beta * C[idx_matrix(ldC, tx,ty)];
        }
        
    // Small kernel to sum: C = alpha *A + beta *B
    __global__ void MatrixAxpyOnGPU_small(data_t *A, data_t *B, data_t *C, int size, int ldA, int ldB, int ldC, data_t alpha, data_t beta)
    {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int gx = tx + blockDim.x * blockIdx.x;
        const int gy = ty + blockDim.y * blockIdx.y;
        
        if(gx >= size || gy >= size) return;
        
        C[idx_matrix(ldB, gx,gy)] = alpha * A[idx_matrix(ldA, gx,gy)] + beta * B[idx_matrix(ldB, gx,gy)];
        
    }   
    
    __global__ void scalarMult_small(data_t *A, data_t *B, int rowA, int ColA, int ldA, 
    int rowB, int ColB, int ldB, data_t scalar)
    {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int gx = tx + blockDim.x * blockIdx.x;
        const int gy = ty + blockDim.y * blockIdx.y;
        
        if(gx >= rowA || gy >= rowB) return ;
        
        B[idx_matrix(ldB, gx,gy)] = scalar * A[idx_matrix(ldA, gx, gy)];
    }
           
} /*namespace linalg*/
            
            