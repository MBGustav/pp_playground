#ifndef _COMMON_CUDA_H_
#define _COMMON_CUDA_H_


#include <stdio.h> 
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>

#include "common.h"



// Check if WidthxHeight is multiple at comp. time
__device__ __host__ constexpr bool isPowerOfTwo(const int number){
    return (number > 0) && ((number - 1) & number) == 0;
}


__device__ __host__ constexpr int floorlog2(unsigned x)
{
    return x == 1 ? 0 : 1+floorlog2(x >> 1);
}

__device__ __host__ constexpr int ceillog2(int x)
{
    return x == 1 ? 0 : floorlog2(x - 1) + 1;
}


__device__ __host__ int matrix_idx(int x, int y)
{
    // Better for GPU and require less operations
    if constexpr (isPowerOfTwo(Width) && isPowerOfTwo(Height))
        return (x + Width) & (Width - 1) + (y + Height) & (Height - 1) << ceillog2(Width);
    // if is not power of two - general case
    return (((x + Width) % Width) + ((y + Height) % Height) * Width);
}

__device__ __host__ int matrix_idx_unsafe(int x, int y)
{
    if constexpr (isPowerOfTwo(Width) && isPowerOfTwo(Height))
        return (x + Width) + (y + Height) << ceillog2(Width);
    // general use-case
    return (((x + Width)) + ((y + Height)) * Width);
}
void DisplayGame(u_char *Universe)
{
    const int w = Width;
    const int h = Height;

    printf("\033[H");
	for_y {
        for_x printf(Universe[idx_matrix1D(y,x)] ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
	fflush(stdout);
}


inline uint64_t _ceil(uint64_t a, uint64_t b){return (a+b-1)/b;}

// GPU TIMER
class Timer {
    float time;
    const uint64_t gpu;
    cudaEvent_t ying, yang;
    std::string version; 
    std::ostream& output;

public:
    Timer (std::string version,std::ostream& output = std::cout, uint64_t gpu=0) : 
            version(version), output(output), gpu(gpu) {
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
    void stop (std::string label, double div=1.0f) {
        cudaSetDevice(gpu);
        cudaEventRecord(yang, 0);
        cudaEventSynchronize(yang);
        cudaEventElapsedTime(&time, ying, yang);
        if(&output == &std::cout)
            std::cout << "TIMING: " << time/div << " ms (" << label << ")" << std::endl;
        else{
            output << version << ", " << label << ", " << time/div << "\n";
            output.flush();  // Certifique-se de sincronizar o arquivo.
        }
    }
};

// ########################################




#endif /*_COMMON_CUDA_H_*/