#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

#include <iostream>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "Matrix.hpp"

// SYCL kernel mangling names
class KernelMultiply;

#define MAX_SIZE_TEMPORARY_BUFFER (512)

void MatrixMultiplication(sycl::queue Q, Matrix &A, Matrix &B, Matrix &C)
{

    // Matrices Size
    const int MatSizeA = A.size();
    const int MatSizeB = B.size();
    const int MatSizeC = C.size();

    const int kArows = B.Col;
    const int kCommon = A.Col;
    const int kBcols = B.Col;


    // Device Pointers
    data_t *dev_A = sycl::malloc_device<data_t>(MatSizeA, Q);
    data_t *dev_B = sycl::malloc_device<data_t>(MatSizeB, Q);
    data_t *dev_C = sycl::malloc_device<data_t>(MatSizeC, Q);

    // Copy data from host to device
    Q.memcpy(dev_A, A.data, MatSizeA * sizeof(data_t)).wait();
    Q.memcpy(dev_B, B.data, MatSizeB * sizeof(data_t)).wait();


    // Using Matricial implementation to multiply Matrices 
    auto matmul_event = Q.single_task<KernelMultiply>([=](){

        data_t c_buffered[MAX_SIZE_TEMPORARY_BUFFER]; 

        //Execute multiplication and use local memory
        for(int i=0; i < kArows; i++){
            for(int j=0; j < kBcols; j++){
                data_t acc = 0.0f;
                for(int k=0; k<kCommon;k++)
                    acc += dev_A[kCommon*i + k] * dev_B[kBcols*k + j];
                c_buffered[kBcols*i +j]  = acc;
            }

        }
        // #pragma unroll
        for(int id=0; id < kArows * kBcols; id++)
            dev_C[id] = c_buffered[id]; 

    });


    //Copy data back from Device
    Q.memcpy(C.data, dev_C, MatSizeC * sizeof(data_t)).wait();
    
    double start = matmul_event.get_profiling_info<
                            sycl::info::event_profiling::command_start>();
    double end = matmul_event.get_profiling_info<
                            sycl::info::event_profiling::command_end>();

    //Convert value from ns, to ms
    std::cout << "Total Duration: " << (double) (end-start) * 1e-6 <<std::endl;
}


#endif /*_KERNEL_HPP_*/