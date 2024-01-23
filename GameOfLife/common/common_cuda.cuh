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


inline int matrix_idx(int x, int y)
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


__global__ void GOLInternalKernel(u_char *Universe, u_char *NewUniverse)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int gx = tx + blockDim.x * blockIdx.x;
	const int gy = ty + blockDim.y * blockIdx.y;
	

	//Out-of-bounds checking (we calc w/ wrap around)
	if(gx > Width && gy > Height)
		return;

	// Stencil to the new universe
    // #TODO: i can improve and reduce idx_matrix1D
	int n = Universe[idx_matrix1D(gy-1,gx-1)] + Universe[idx_matrix1D(gy-1,gx)] + Universe[idx_matrix1D(gy-1,gx+1)] +
			Universe[idx_matrix1D(gy  ,gx-1)] +                                   Universe[idx_matrix1D(gy  ,gx+1)] +
			Universe[idx_matrix1D(gy+1,gx-1)] + Universe[idx_matrix1D(gy+1,gx)] + Universe[idx_matrix1D(gy+1,gx+1)];

	//New Universe is written
	NewUniverse[idx_matrix1D(gy, gx)] = ((n | Universe[idx_matrix1D(gy,gx)]) == 3 );
}


void GameOfLifeKernel(u_char *Universe, int NumberOfGenerations)
{
    srand(777);
    //Game parameters
    constexpr int w= Width;
	constexpr int h= Height;
	const int kSizeUniv = Width * Height; 
	const int kBinSize = kSizeUniv * sizeof(u_char);
	
    // GPU Parameters
    constexpr int nx_threads= THREADS_X;
	constexpr int ny_threads= THREADS_Y;

    
    for_xy Universe[idx_matrix1D(y, x)] = rand() < RAND_MAX / 10 ? 1 : 0;
    // for_xy Universe[idx_matrix1D(y, x)] = y == x ? 1 : 0;


	u_char *UniverseGPU, *NewUniverseGPU;
	cudaMalloc((void**)& UniverseGPU, kBinSize);
	cudaMalloc((void**)& NewUniverseGPU, kBinSize);

	cudaMemcpy(UniverseGPU, Universe, kBinSize, cudaMemcpyHostToDevice);

	dim3 blocks(nx_threads, ny_threads);
	dim3 grids(_ceil(Width,nx_threads), _ceil(Height, ny_threads));
	// dim3 grids_border(_ceil(2,nx_threads), _ceil(2, ny_threads));

	//deal with center
    while(NumberOfGenerations--){
        GOLInternalKernel<<<grids, blocks>>>(UniverseGPU, NewUniverseGPU);
    
        swap_pointers((void**)& UniverseGPU,(void**)& NewUniverseGPU);
        // u_char *temp = UniverseGPU;
        // UniverseGPU = NewUniverseGPU;
        // NewUniverseGPU = temp;
        
        #ifdef _DEBUG_PER_STEP
            cudaMemcpy(Universe, UniverseGPU, kBinSize, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            DisplayGame(Universe);
            getchar();
        #endif /*_DEBUG_PER_STEP*/

    }

	//copy back to host
	cudaMemcpy(Universe, UniverseGPU, kBinSize, cudaMemcpyDeviceToHost);

    DisplayGame(Universe);
}




// ########################################




#endif /*_COMMON_CUDA_H_*/