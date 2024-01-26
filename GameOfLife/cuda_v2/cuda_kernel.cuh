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

/*
E se tivermos um Universe of Life ? 
Vamos implementar um Game of Life que seja suficientemente grande!
Usaremos streaming...
 */
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
	constexpr int NumStreams= NUM_STREAMS;
	constexpr int chunk_size = _ceil(kSizeUniv, NumStreams);
	cudaStream_t streams[NumStreams];
	dim3 blocks(nx_threads, ny_threads);
	dim3 grids(20,20);

	


    // Que tal usar cuRAND neste exemplo?
    for_xy Universe[idx_matrix1D(y, x)] = rand() < RAND_MAX / 10 ? 1 : 0;
    


	u_char *UniverseGPU, *NewUniverseGPU;
	cudaMalloc((void**)& UniverseGPU, kBinSize);
	cudaMalloc((void**)& NewUniverseGPU, kBinSize);

	// Podemos fazer assincrono?
	cudaMemcpy(UniverseGPU, Universe, kBinSize, cudaMemcpyHostToDevice);

	// Criamos os Streams
	for(int stream=0; stream < NumStreams; stream++)
		cudaStreamCreate(&streams[stream]);
	
	while(NumOfGenerations--)
	{
		// Submetemos os dados para os streams
		for(int stream=0; stream < NumStreams; stream++){
			int lower = chunk_size * stream; //posicao de offset na memoria
			int upper = min(lower + chunk_size, kSizeUniv); // Ponteiro para o final da janela
			int width = upper - lower; 
			
			// cudaMemcpyAsync(UniverseGPU + lower,   // ptr. para gpu
			// 				Universe    + lower,   // ptr. para cpu
			// 				sizeof(u_char)*width,  // Tam. dos dados
			// 				cudaMemcpyHostToDevice,// Sentido de copia 
			// 				streams[stream]);	   // qual fila de stream
							
			// Emissao de Kernel <<<grid, block, Mem, #Stream>>>
			GOLInternalKernel<<<grids, blocks, 0, streams[stream]>>>
			(UniverseGPU + lower, NewUniverseGPU + lower);
		}

		// Sincronizamos os Streams (eles sao assincronos!)
		for(int stream=0; stream < NumStreams; stream++)
			cudaStreamSynchronize(streams[stream]);

			// Swap ocorre apenas depois de calculada toda a geração
		swap_pointers((void**)& UniverseGPU,(void**)& NewUniverseGPU);
	
	#ifdef _DEBUG_PER_STEP
		cudaMemcpy(Universe, UniverseGPU, kBinSize, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		DisplayGame(Universe);
		getchar();
	#endif /*_DEBUG_PER_STEP*/	
	}


	
	// Copiamos diretamente para o HOST
	cudaMemcpy(Universe, UniverseGPU, kBinSize, cudaMemcpyDeviceToHost);
	
	// Necessario ?
	cudaDeviceSynchronize();
	
	DisplayGame(Universe);
	
	
	
	
	// Destruimos os Streams
	for(int stream=0; stream < NumStreams; stream++)
		cudaStreamDestroy(&streams[stream]);
	
	// Liberamos a memoria da GPU
	cudaFree(UniverseGPU);
	cudaFree(NewUniverseGPU);
}



#endif  /*_CUDA_KERNEL_H_*/