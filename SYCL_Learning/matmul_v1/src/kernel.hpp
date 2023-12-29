#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

#include <iostream>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "Matrix.hpp"

// SYCL kernel mangling names
class KernelMultiply;

#define MAX_SIZE_TEMPORARY_BUFFER (24)

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

        data_t copy_of_dprod[MAX_SIZE_TEMPORARY_BUFFER+1];
        
        //Execute multiplication and use local memory
        [[intel::ivdep]]
        for (int idx = 0; idx < kArows * kBcols; ++idx) {
            int i = (int) idx / kBcols;
            int j = (int) idx % kBcols;

            //Initialize shift register with zeros 
            #pragma unroll
            for(int i= 0; i < MAX_SIZE_TEMPORARY_BUFFER; i++)
                copy_of_dprod[i] = 0.0f;
            
            // Dot Product: A[i, k] * B[k, j]
            for (int k = 0; k < kCommon; k++) 
            {
                //Calculate the product btwn A & B (pipeline)
                data_t current_dprod = dev_A[kCommon * i + k] * dev_B[kBcols * k + j];

                copy_of_dprod[MAX_SIZE_TEMPORARY_BUFFER] = copy_of_dprod[0] + current_dprod;

                //Shift Register (from right to left)
                #pragma unroll (MAX_SIZE_TEMPORARY_BUFFER)
                for(int pip= 0; pip < MAX_SIZE_TEMPORARY_BUFFER; pip ++)
                    copy_of_dprod[pip] = copy_of_dprod[pip+1];
            } //End of dot-prod

            data_t acc = 0.0f;
            #pragma unroll (MAX_SIZE_TEMPORARY_BUFFER)
            for(int ii=0; ii < MAX_SIZE_TEMPORARY_BUFFER; ii++)
                acc += copy_of_dprod[ii];

            dev_C[i*kArows + j] = acc;
        }
 
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