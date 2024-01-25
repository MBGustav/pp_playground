#include <cassert>
// #include <cuda_runtime.h>

#include "cudaMatrix.cuh"


__global__ void kernel() {
    int tid = threadIdx.x;
    printf("%d\n",(tid));
}
namespace gpuLinalg{

    __global__ void MatrixMultiplicationOnGPU(data_t *A, data_t *B, data_t *C, int m, int k, int n, int ldA, int ldB, int ldC, data_t alpha, data_t beta)
    {
        // Local Range (memo-access)
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        //global range
        int gx = tx + blockDim.x * blockIdx.x;
        int gy = ty + blockDim.y * blockIdx.y;
        
        if(gx > m || gx > n) return;
        
        __shared__ data_t SharedMemoA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ data_t SharedMemoB[BLOCK_SIZE][BLOCK_SIZE];


        // if(gx == 0 && gy == 0) {
        //     printf("Matrix on GPU \n");
        //     for(int i = 0; i < m; i++){
        //         for(int j = 0; j < m; j++)
        //             printf("%f ",A[idx_matrix(ldA, i,j)]);
        //         printf("\n");
        //     }
        //     printf("\n\n");

        // }

        //copy from global to local
        data_t acc = 0.0f;
    for (int stride_k = 0; stride_k < k; stride_k += BLOCK_SIZE)
    {
        // Carregar submatriz da matriz A para a memória local
        SharedMemoA[ty][tx] = A[idx_matrix(ldA, gx, stride_k + tx)];

        // Carregar submatriz da matriz B para a memória local com stride
        SharedMemoB[ty][tx] = B[idx_matrix(ldB, stride_k + ty, gy)];

        // Sincronizar threads antes de calcular a próxima submatriz
        __syncthreads();

        // Computar o produto da submatriz
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            acc += SharedMemoA[ty][i] * SharedMemoB[i][tx];
        }

        // Sincronizar threads antes de carregar a próxima submatriz
        __syncthreads();
    }
    C[idx_matrix(ldC,gx, gy)] = alpha * acc + beta * C[idx_matrix(ldC,gx, gy)];
        // __syncthreads();
    }
} /*namespace linalg*/


// template <typename T>
// __global__ void MatrixMultiplicationWithStride(data_t A, data_t B, data_t C, data_t alpha, data_t beta)
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