#ifndef _COMMON_CUDA_H_
#define _COMMON_CUDA_H_


#include <stdio.h> 
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "common.h"


// Check if WidthxHeight is multiple at comp. time
constexpr bool isPowerOfTwo(const int number){
    return (number > 0) && ((number - 1) & number) == 0;
}

int matrix_idx(int x, int y)
{
    // Better for GPU and require less operations
    if constexpr (isPowerOfTwo(Width) && isPowerOfTwo(Height))
    return (x + Width) & (Width - 1) + (y + Height) & (Height - 1) * Width;
    
    // if is not power of two - general case
    return (((x + Width) % Width) + ((y + Height) % Height) * Width);
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

// ########################################




#endif /*_COMMON_CUDA_H_*/