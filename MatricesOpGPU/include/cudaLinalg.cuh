#ifndef _CUDA_LINALG_H_
#define _CUDA_LINALG_H_

#include "cuda_common.cuh"
#include "cudaMatrix.cuh"
#include <vector>


/*
Realizes an Matrix Multiplication: 
    input: A, B [matrix]
    input: alpha, beta [scala]
    output C
    This operation realizes the following operation:
    C = alpha (A x B) + beta C
    // template<typename T>
*/ 
namespace gpuLinalg{
    void MatMul(cudaMatrix &A, cudaMatrix &B, cudaMatrix &C,data_t alpha, data_t beta);
    void MatMulOnCPU(cudaMatrix &A, cudaMatrix &B, cudaMatrix &C,data_t alpha, data_t beta);
    
    /* B = scalar * A*/
    void scalarMult(cudaMatrix &A, cudaMatrix &B, data_t scalar);
    void MatAdd(cudaMatrix &A, cudaMatrix &B, cudaMatrix &C);
    bool EqualMatrices(cudaMatrix &A, cudaMatrix &B);
    
    // viability offload checkers
    offload_t matmul_offload(int A_size, int B_size);
}





#endif /*_CUDA_LINALG_H_*/