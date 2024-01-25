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
    
}





#endif /*_CUDA_LINALG_H_*/