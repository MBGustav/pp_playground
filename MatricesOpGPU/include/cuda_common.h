#ifndef __CUDA_COMMON_H__
#define __CUDA_COMMON_H__


#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

//GPU PARAMETERS
#define MINIMUM_DISPLAY (6)


#ifndef Thread_x
#define Thread_x (1<<5)
#endif
#ifndef Thread_y
#define Thread_y (1<<5)
#endif

#ifndef num_threads
#define num_threads (Thread_x * Thread_y)
#endif


// DATA TYPE DEFINITION
#if defined DATA_PRECISION == 1
#define data_t double
#else
#define data_t float
#endif


//================ PERFORMANCE PADDING ================  
// Working with a row-major cases.

#if DATA_PRECISION == 1
#define LD_ALIGN 256
#define LD_BIAS 8
#else
#define LD_ALIGN 512
#define LD_BIAS 16
#endif

#define HPL_PTR(ptr_, al_) ((((size_t)(ptr_) + (al_)-1) / (al_)) * (al_))

// #define MALLOC(x) (data_t *)mkl_malloc((x), 64)
#define MALLOC(x, size) x = (data_t*)malloc((size) * sizeof(size))

// Setting two cases: transposed and non-transposed ?
// TODO: Is there a way to use for both cases  -> taking into consideration performance..
#define idx_matrix(i, j, ld) (((j) * (ld)) + (i))
#define idx_matrix_trp(i, j, ld, transposed) ((transposed) ? ((j) * (ld) + (i)) : ((i) * (ld) + (j)))

// PADDING CONFIGURATION -> check for GPU
#if defined(PAD_LD)
static inline int getld(int x) {
  int ld;
  ld = HPL_PTR(x, LD_ALIGN); // Rule 1
  if (ld - LD_BIAS >= x)
    ld -= LD_BIAS;
  else
    ld += LD_BIAS; // Rule 2
  return ld;
}
#else
static inline int getld(int x) { return x; }
#endif
//=====================================================  




// GPU ERROR CHECKER
inline void check_last_error ()
{
    cudaError_t err;
    if ((err = cudaGetLastError()) != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "
                  << __FILE__ << ", line " << __LINE__ << std::endl;
            exit(1);
    }
}






inline uint64_t sdiv (uint64_t a, uint64_t b){return (a+b-1)/b;}

typedef enum ChangeHandler{
    Equal,
    ChangeOnHost,
    ChangeOnDevice
}ChangeHandler;

typedef enum OffloadSelect
{
    CPU,
    GPU_01,
    GPU_02
}OffloadSelect;


// GPU TIMER
class Timer {
    float time;
    const uint64_t gpu;
    cudaEvent_t ying, yang;
public:
    Timer (uint64_t gpu=0) : gpu(gpu) {
        cudaSetDevice(gpu);
        cudaEventCreate(&ying);
        cudaEventCreate(&yang);
    }

    ~Timer ( ) {
        cudaSetDevice(gpu);
        cudaEventDestroy(ying);
        cudaEventDestroy(yang);
    }
    void start ( ) {
        cudaSetDevice(gpu);
        cudaEventRecord(ying, 0);
    }
    void stop (std::string label) {
        cudaSetDevice(gpu);
        cudaEventRecord(yang, 0);
        cudaEventSynchronize(yang);
        cudaEventElapsedTime(&time, ying, yang);
        std::cout << "TIMING: " << time << " ms (" << label << ")" << std::endl;
    }
};



#endif /*__CUDA_COMMON_H__*/
