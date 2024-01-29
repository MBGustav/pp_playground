#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_


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

	Timer gpu_timer(0);


    for_xy Universe[idx_matrix1D(y, x)] = rand() < RAND_MAX / 10 ? 1 : 0;
    // for_xy Universe[idx_matrix1D(y, x)] = y == x ? 1 : 0;

	dim3 blocks(nx_threads, ny_threads);
	dim3 grids(_ceil(Width,nx_threads), _ceil(Height, ny_threads));
	// dim3 grids_border(_ceil(2,nx_threads), _ceil(2, ny_threads));

	u_char *UniverseGPU, *NewUniverseGPU;
	gpu_timer.start();
	cudaMalloc((void**)& UniverseGPU, kBinSize);
	cudaMalloc((void**)& NewUniverseGPU, kBinSize);
	gpu_timer.stop("Alloc Vectors");

	gpu_timer.start();
	cudaMemcpy(UniverseGPU, Universe, kBinSize, cudaMemcpyHostToDevice);
	gpu_timer.stop("Copy Memory H->D");


	gpu_timer.start();
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
	gpu_timer.stop("All Kernels executed");


	//copy back to host
	gpu_timer.start();
	cudaMemcpy(Universe, UniverseGPU, kBinSize, cudaMemcpyDeviceToHost);
	gpu_timer.stop("Copy Memory D->H");

    	
	#ifdef  _DISPLAY_GAME
	DisplayGame(Universe);
	#endif/*_DISPLAY_GAME */
	
}



#endif  /*_CUDA_KERNEL_H_*/