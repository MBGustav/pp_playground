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
#define Thread_x (32)
#endif
#ifndef Thread_y
#define Thread_y (32)
#endif

#define BLOCK_SIZE (32)

#ifndef num_threads
#define num_threads (Thread_x * Thread_y)
#endif

#define EPSILON (0.05f)


#define float_equal(x, y) (fabs((x) - (y)) < EPSILON ? 1: 0)
                            //(fabs((x) - (y)) <= EPSILON ? 1 : 0 )


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
#define idx_matrix(ld, i, j) (((j) * (ld)) + (i))
#define idx_matrix_t(ld, i, j) (((i) * (ld)) + (j))


// PADDING CONFIGURATION -> check for GPU
#if defined(PAD_LD)
static inline int ld_padding(int x) {
  int ld;
  ld = HPL_PTR(x, LD_ALIGN); // Rule 1
  if (ld - LD_BIAS >= x)
    ld -= LD_BIAS;
  else
    ld += LD_BIAS; // Rule 2
  return ld;
}
#else
static inline int ld_padding(int x) { return x; }
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


/* enum that deals with offload  type*/
typedef enum offload_t
{
    cpu,
    gpu_easy,
    gpu_hard
}offload_t;

/*Enum to deal with modifications between Host <-> Device*/
typedef enum ChangeHandler{
    Equal, 
    ChangeOnHost,
    ChangeOnDevice
}ChangeHandler;


class TimerCPU{
private:
    float time;
    long ying, yang;

public:
    TimerCPU(): time(0){ } 
    ~TimerCPU(){ }

    void start(){
        struct timeval tm;
        if(gettimeofday( &time, 0 )) exit(-1);
        ying = 1000000 * time.tv_sec + time.tv_usec;
    } 
        
    float stop ( std::string label, bool output=false ){
        struct timeval time;
        if(gettimeofday( &time, 0 )) exit(-1);
        yang = 1000000 * time.tv_sec + time.tv_usec;
        float time = (yang - ying) / 1000.0;
        if(output)
            std::cout << "TIME(CPU): " <<
            time << " ms (" << label << ")" << std::endl;

        return time;
    } 


};


//#TODO: include flags to (not) compile this class
// GPU TIMER
class TimerGPU {
    float time;
    const uint64_t gpu;
    cudaEvent_t ying, yang;
public:
    TimerGPU (uint64_t gpu=0) : gpu(gpu) {
        cudaSetDevice(gpu);
        cudaEventCreate(&ying);
        cudaEventCreate(&yang);
    }

    ~TimerGPU ( ) {
        cudaSetDevice(gpu);
        cudaEventDestroy(ying);
        cudaEventDestroy(yang);
    }
    void start ( ) {
        cudaSetDevice(gpu);
        cudaEventRecord(ying, 0);
    }
    float stop ( std::string label, bool output=false ) {
        cudaSetDevice(gpu);
        cudaEventRecord(yang, 0);
        cudaEventSynchronize(yang);
        cudaEventElapsedTime(&time, ying, yang);
        if(output) 
            std::cout << "TIME(GPU): " << 
            time << " ms (" << label << ")" << std::endl;

        return time;
    }
};



#endif /*__CUDA_COMMON_H__*/
