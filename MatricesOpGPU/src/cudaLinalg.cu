#include <cuda_runtime.h>
#include <cassert>
#include "cudaLinalg.cuh"
#include "cudaMatrix.cuh"
// #include "cudaLinalg.cu"
#include "kernel.cu"

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
        
        MatrixMultiplicationOnGPU<<<grid, blocks>>>(A.dataGPU(), B.dataGPU(), C.dataGPU(),
                                                    A.getRow(), A.getCol(), B.getCol(), 
                                                    A.get_ld(), B.get_ld(), C.get_ld(), alpha, beta);
        
        C.changeOccurred(ChangeOnDevice);
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