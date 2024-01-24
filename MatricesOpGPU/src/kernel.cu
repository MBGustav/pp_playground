#include <cassert>
#include <cuda_runtime.h>

#include "cudaMatrix.h"



namespace gpuLinalg{

    __global__ void MatrixMultiplicationOnGPU(cudaMatrix &A, cudaMatrix &B, cudaMatrix &C, data_t alpha, data_t beta)
    {
        // Local Range (memo-access)
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        //global range
        int gx = tx + blockDim.x * blockIdx.x;
        int gy = ty + blockDim.y * blockIdx.y;

        
        __shared__ data_t SharedMemoA[blockDim.x][blockDim.y];
        __shared__ data_t SharedMemoB[blockDim.x][blockDim.y];
        __shared__ data_t SharedMemoC[blockDim.x][blockDim.y];


        //copy from global to local
        SharedMemo[tx][ty] = A.at(gx, gy);
        SharedMemo[tx][ty] = B.at(gx, gy);

        __syncthreads();

        // Time to multiply
        for(int stride_i = gx; stride_i < A.getRow(); stride_i+=gridDim.x*blockDim.x)
        for(int stride_j = gy; stride_j < B.getCol(); stride_j+=gridDim.y*blockDim.y)
        {
            data_t acc = 0.0f;
            for(int kk=0; kk < blockDim.x; kk++)
                // acc += A(i, k) * B(k, j)
                acc += SharedMemoA[tx][kk] * SharedMemo[kk][ty];
            SharedMemoC[tx][ty] = alpha * acc;
        }

        __syncthreads();

        C.at(gx.gy) = C.at(gx,gy) * beta + SharedMemoC[tx][ty];
        


    }
} /*namespace linalg*/


// template <typename T>
// __global__ void MatrixMultiplicationWithStride(cudaMatrix A, cudaMatrix B, cudaMatrix C, data_t alpha, data_t beta)
// {
//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int blockDimX = blockDim.x;
//     const int blockDimY = blockDim.y;
//     const int gridDimX = gridDim.x;

//     // Tamanho do bloco para processar (matriz de bloco)
//     const int blockRowStart = by * blockDimY;
//     const int blockColStart = bx * blockDimX;

//     // Índices globais
//     const int row = blockRowStart + ty;
//     const int col = blockColStart + tx;

//     // Memória local para submatrizes
//     __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

//     // Inicialização do acumulador
//     T acc = 0.0f;

//     // Loop sobre as submatrizes
//     for (int k = 0; k < A.getCol(); k += BLOCK_SIZE)
//     {
//         // Carregar submatriz da matriz A para a memória local
//         As[ty][tx] = A.at(row, k + tx);

//         // Carregar submatriz da matriz B para a memória local com stride
//         Bs[ty][tx] = B.at(k + ty, col);

//         // Sincronizar threads antes de calcular a próxima submatriz
//         __syncthreads();

//         // Computar o produto da submatriz
//         for (int i = 0; i < BLOCK_SIZE; ++i)
//         {
//             acc += As[ty][i] * Bs[i][tx];
//         }

//         // Sincronizar threads antes de carregar a próxima submatriz
//         __syncthreads();
//     }

//     // Atualizar a matriz de resultado
//     C.at(row, col) = alpha * acc + beta * C.at(row, col);
// }