#include <cassert>
#include <cuda_runtime.h>

#include "cudaMatrix.h"
#include "cudaLinalg.h"
// #include "cudaLinalg.cu"
// #include "LinalgKernels.cu"

namespace gpuLinalg{

    constexpr int MatMul_MinimumSize = num_threads;

    static inline void MatrixMultiplicationOnCPU(cudaMatrix &A, cudaMatrix &B, cudaMatrix &C,data_t alpha, data_t beta){
        #pragma omp parallel for reduction(+:acc)
        for (int i = 0; i < A.getRow(); i++) {
            for (int j = 0; j < B.getCol(); j++) {
                data_t acc=0.0f;
                for (int k = 0; k < B.getRow(); k++)
                    acc += A.at(i, k) * B.at(k, j);
                C.at(i, j) = acc;
            }
        }

    }
    
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
            for(int kk=0; k < blockDim.x; kk++)
                // acc += A(i, k) * B(k, j)
                acc += SharedMemoA[tx][kk] * SharedMemo[kk][ty];
            SharedMemoC[tx][ty] = alpha * acc;
        }

        __syncthreads();

        C.at(gx.gy) = C.at(gx,gy) * beta + SharedMemoC[tx][ty];
        


    }

    
    // Execute matricial operation: C  = alpha . A x B + beta . C
    // template<typename T>
    void MatMul(cudaMatrix &A, cudaMatrix &B, cudaMatrix &C,data_t alpha, data_t beta)
    {   
        /*some cases are not necessary to deal with it all the time*/
        #ifdef ASSERT_MATRIX_PARAMETERS 
        assert(A.getCol() == B.getRow());
        assert(C.getCol() == A.getCol() && C.getRow() == A.getRow());
        #endif /*ASSERT_MATRIX_PARAMETERS*/

        const int Size = C.getSize();  
        if(MatMul_MinimumSize < Size)
            MatrixMultiplicationOnCPU(A, B, C,alpha, beta);
        
        /* Declare size of grid and blocks */
        dim3 blocks(Thread_x,Thread_y);
        dim3 grid(sdiv(C.getRow(), blocks.x), sdiv(C.getCol(), blocks.y));
        
        MatrixMultiplicationOnGPU<<<grid, blocks>>>(A, B, C, alpha, beta);
    }

    
    // static kernel declararion to use only inside this operation. (kernel gpu)
    // template<typename T> //#TODO: I can do it ?

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