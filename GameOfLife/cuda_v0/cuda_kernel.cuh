#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_

#include <fstream>

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


void GameOfLifeKernel(u_char *Universe,const int NumberOfGenerations)
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
	u_char *UniverseGPU, *NewUniverseGPU;
	dim3 blocks(nx_threads, ny_threads);
	dim3 grids(_ceil(Width,nx_threads), _ceil(Height, ny_threads));
	
	const std::string label = "cuda_v0," +
							  std::to_string(NumberOfGenerations) + "," + 
							  std::to_string(w);

	#ifdef _DISPLAY_DATA_PROFILING
	Timer general_timer("", std::cout);
	#else
	std::ofstream file_output("bench_AllSteps.txt", std::ios_base::app);
	Timer general_timer(label, file_output);
	#endif

	//timer para comparar tempo entre execucoes
	std::ofstream file_bench("benchmark.txt", std::ios_base::app);
	Timer comp_timer(label, file_bench);
	

    for_xy Universe[idx_matrix1D(y, x)] = rand() < RAND_MAX / 10 ? 1 : 0;
    
	

	general_timer.start();
	cudaMalloc((void**)& UniverseGPU, kBinSize);
	cudaMalloc((void**)& NewUniverseGPU, kBinSize);
	general_timer.stop("Alloc Vectors");
	
	general_timer.start();
	comp_timer.start();
	cudaMemcpy(UniverseGPU, Universe, kBinSize, cudaMemcpyHostToDevice);
	general_timer.stop("Copy Memory H->D");


	// general_timer.start();
	
	int Generation = NumberOfGenerations;
    while(Generation--)
	{
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
	general_timer.stop("Avg. per Kernel", static_cast<double>(NumberOfGenerations));
	//copy back to host
	general_timer.start();
	cudaMemcpy(Universe, UniverseGPU, kBinSize, cudaMemcpyDeviceToHost);
	comp_timer.stop("Total Time Execution");
	general_timer.stop("Copy Memory D->H");
	
    	
	#ifdef  _DISPLAY_GAME
	DisplayGame(Universe);
	#endif/*_DISPLAY_GAME */
	
}



#endif  /*_CUDA_KERNEL_H_*/