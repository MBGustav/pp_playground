#include <cuda_runtime.h>
#include <cassert>
#include "cuda_common.cuh"
#include "cudaLinalg.cuh"
#include "cudaMatrix.cuh"
// #include "cudaLinalg.cu"
#include "kernel.cu"

namespace gpuLinalg{
    
    offload_t matmul_offload(int A_size, int B_size)
    {
        constexpr int min_size = num_threads;
        const int avg_items = (A_size + B_size)>>1;
        
        if(min_size < avg_items) return cpu;
        
        if( min_size < 4*avg_items ) return gpu_easy;
        
        return gpu_hard;
    }

    // TODO: check when is viable to offload to GPU
    offload_t simple_offload(int A_size)
    {
        constexpr int min_size = num_threads;
        if(min_size <= A_size) return cpu;
        
        if(min_size < 4*A_size) return gpu_easy;
        
        return gpu_hard;
    }
    
    
    void MatMulOnCPU(cudaMatrix &A, cudaMatrix &B,
        cudaMatrix &C,data_t alpha, data_t beta)
    {
        // #pragma omp parallel for reduction(+:acc)
        for (int i = 0; i < A.getRow(); i++) {
            for (int j = 0; j < B.getCol(); j++) {
                data_t acc=0.0f;
                for (int k = 0; k < B.getRow(); k++)
                acc += A.at(i, k) * B.at(k, j);
                C.at(i, j) = beta * C.at(i, j) + alpha * acc;
            }
        }
    }
    
    void scalarMultOnCPU(cudaMatrix &A, cudaMatrix &B, data_t scalar){
        const int Arow = A.getRow();
        const int Acol = A.getCol();
        
        #pragma omp parallel for
        for (int i = 0; i < A.getRow(); i++) 
        for (int j = 0; j < B.getCol(); j++)
        B.at(i, j) = scalar * A.at(i, j);
        
    }       
    
    // Execute matricial operation: C  = alpha . A x B + beta . C
    // template<typename T>
    void MatMul(cudaMatrix &A, cudaMatrix &B, cudaMatrix &C,data_t alpha, data_t beta)
    {   
        /*some cases are not necessary to deal with it all the time*/
        #ifndef ASSERT_MATRIX_PARAMETERS 
        assert(A.getCol() == B.getRow());
        assert(C.getRow() == A.getRow() && C.getCol() == B.getCol());
        #endif /*ASSERT_MATRIX_PARAMETERS*/

        const int Size = C.getSize();  
        
        dim3 blocks(Thread_x,Thread_y);
        dim3 grid(sdiv(C.getRow(), blocks.x), sdiv(C.getCol(), blocks.y));
        
        
        offload_t offload = matmul_offload(A.getSize(), B.getSize());
        if(offload == cpu) MatMulOnCPU(A, B , C, alpha, beta);
        else if(offload == gpu_easy)
        MatrixMultiplicationOnGPU_small<<<grid, blocks>>>(
            A.dataGPU(), B.dataGPU(), C.dataGPU(),
            A.getRow(), A.getCol(), B.getCol(), 
            A.get_ld(), B.get_ld(), C.get_ld(), alpha, beta);
        // else gpu_hard
        else
        MatrixMultiplicationOnGPU<<<grid, blocks>>>(
            A.dataGPU(), B.dataGPU(), C.dataGPU(),
            A.getRow(), A.getCol(), B.getCol(), 
            A.get_ld(), B.get_ld(), C.get_ld(), alpha, beta);
        
        C.changeOccurred(offload == cpu ? ChangeOnHost : ChangeOnDevice);
    }
    
    /* B =  scalar * A */
    void scalarMult(cudaMatrix &A, cudaMatrix &B, data_t scalar)
    {
        
        #ifndef ASSERT_MATRIX_PARAMETERS 
        assert(A.getCol() == B.getCol() && A.getRow() == B.getRow());
        #endif /*ASSERT_MATRIX_PARAMETERS*/
        
        const int size = B.getSize();
        dim3 blocks(Thread_x, Thread_y);
        dim3 grid(sdiv(C.getRow(), blocks.x), sdiv(C.getCol(), blocks.y));
        
        offload_t offload  = simple_offload(size);
        
        if(offload == cpu){
            scalarMultOnCPU(A, B, scalar);
        }else {
            scalarMult_small<<<grid, blocks>>>(
                A.data(), B.data(), A.getRow(), A.getCol(), A.get_ld(), 
                B.getRow(), B.getCol(), B.get_ld(), scalar);                
        }
        
        B.changeOccurred(offload == cpu ? ChangeOnHost : ChangeOnDevice);
        
        return;
    }
    
    void MatAdd(cudaMatrix &A, cudaMatrix &B, cudaMatrix &C)
    {
        #ifndef ASSERT_MATRIX_PARAMETERS 
        assert(A.getCol() == B.getCol() && A.getRow() == B.getRow());
        assert(C.getCol() == B.getCol() && C.getRow() == B.getRow());
        #endif /*ASSERT_MATRIX_PARAMETERS*/
        
        dim3 blocks(Thread_x, Thread_y);
        dim3 grid(sdiv(C.getRow(), blocks.x), sdiv(C.getCol(), blocks.y));
        // std::cout << grid.x << ", " << grid.y << std::endl;
        
        MatrixAxpyOnGPU_small<<<grid, blocks>>>(A.dataGPU(), B.dataGPU(), C.dataGPU(), A.getSize(), 
        A.get_ld(), B.get_ld(), C.get_ld(), 1.0, 1.0);
        // sumMatrix_small<<<grid, blocks>>>(A.dataGPU(), B.dataGPU(), C.dataGPU(), A.getRow(), A.getCol(), A.get_ld(), B.get_ld(), C.get_ld());
        C.changeOccurred(ChangeOnDevice);
    }
    // static kernel declararion to use only inside this operation. (kernel gpu)
    
    
    bool EqualMatrices(cudaMatrix &A, cudaMatrix &B) {
        assert(A.getCol() == B.getCol() && A.getRow() == B.getRow());
        
        const int Row = A.getRow();
        const int Col = A.getCol();
        bool ans = true;
        
        // To do Locally, we must synch values
        A.SynchronizeValues(); 
        B.SynchronizeValues();
        
        #pragma omp parallel for
        for (int r = 0; r < Row; r++) {
            for (int c = 0; c < Col; c++) {
                if (!float_equal(A.const_at(r, c), B.const_at(r, c))) {
                    // std::cout << "Difference at: (" << r << "," << c << "): " << A.at(r,c) << " != " << B.at(r,c) << std::endl;
                    return false;
                }
            }
        }
        return true; // Matrices are equal
    }
    
        
} /*namespace linalg*/
    