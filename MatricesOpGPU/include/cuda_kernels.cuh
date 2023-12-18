#ifndef _CUDA_KERNELS_CUH_
#define _CUDA_KERNELS_CUH_

#include "cudaMatrix.h"
#include "cuda_common.h"


// More  elaborated -> maybe I can use it in general cases ?
Wrapper_matmul01(cudaMatrix &A,cudaMatrix &B,cudaMatrix &C,double alpha, double beta)
{
	// A[m, k] x B[k, n] = C[m, k]

	int m = A.getRow();
	int k = A.getCol();
	int n = A.getRow();

	//using width to select total elements
	dim3 threadsPerBlock(Thread_x, Thread_y);
	dim3 NumBlocks(m/Thread_x, k/Thread_y);


	kernel_matmul<<<NumBlocks, threadsPerBlock>>>(A,B,C,alpha, beta);

}

// Most simple, due to the quantity items
Wrapper_matmul01(data_t *A,data_t *B,data_t *C,double alpha, double beta,bool transA,bool transB)
{



}


__global__ kernel_matmul_simple(data_t *A,data_t *B,data_t *C,double alpha, double beta,bool transA,bool transB, int kk)
{
	data_t *Av = A.getDataGPU();
	data_t *Bv = A.getDataGPU();
	data_t *Cv = A.getDataGPU();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	data_t acc = 0.0f;
	for(int k=0; k < kk ;k++)
		acc += Av[idx_matrix_trp(i,k,transA)] * B[idx_matrix_trp(k,j,transB)];
	Cv[i,j] = acc;

}


#endif //_CUDA_KERNELS_CUH_