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
	constexpr int NumStreams= NUM_STREAMS;
	u_char *UniverseGPU, *NewUniverseGPU;
	const int chunk_size = _ceil(kSizeUniv, NumStreams);
	cudaStream_t streams[NumStreams];
	dim3 blocks(nx_threads, ny_threads);
	dim3 grids(20,20);

	const std::string label = "cuda_v2," +
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
	
    // Que tal usar cuRAND neste exemplo?
    for_xy Universe[idx_matrix1D(y, x)] = rand() < RAND_MAX / 10 ? 1 : 0;
    
	general_timer.start();
	cudaMalloc((void**)& UniverseGPU, kBinSize);
	cudaMalloc((void**)& NewUniverseGPU, kBinSize);
	general_timer.stop("Alloc Vectors");
	
	// Podemos fazer assincrono? Muito Complexo!
	general_timer.start();
	comp_timer.start();
	cudaMemcpy(UniverseGPU, Universe, kBinSize, cudaMemcpyHostToDevice);
	general_timer.stop("Copy Memory H->D");
	// Criamos os Streams
	for(int stream=0; stream < NumStreams; stream++)
		cudaStreamCreate(&streams[stream]);
	
	general_timer.start();
	for(int Generation=0; Generation <  NumberOfGenerations; Generation++)
	{
		// Submetemos os dados para os streams
		for(int stream=0; stream < NumStreams; stream++)
		{
			int lower = chunk_size * stream; //posicao de offset na memoria
			int upper = min(lower + chunk_size, kSizeUniv); // Ponteiro para o final da janela
			int width = upper - lower;

			//Primeira Geração copiamos os dados para GPU:
			/* 	 Caso 1: 					Caso 2:
				+---------+                 +---------+
				|x|     |x|                 |  |xxx|  |
				|         |                 |x|     |x|
				|x|     |x|                 |  |xxx|  |
				+---------+                 +---------+
				1.   Como temos contorno de borda, existe a necessidade de realizar 
					 previamente 4 copias, que contém as bordas do jogo. 
				
				2. Controle de Bordas sem canto.
				if(Generation == 0) {
				cudaMemcpyAsync(UniverseGPU + lower,   // ptr. para gpu
								Universe    + lower,   // ptr. para cpu
								sizeof(u_char)*width,  // Tam. dos dados
								cudaMemcpyHostToDevice,// Sentido de copia 
								streams[stream]);	   // qual fila de stream
				
				// Para os casos de borda, necessario mais duas copias!
				if(is_lower_border())
				{
					
					cudaMemcpyAsync(UniverseGPU +(kSizeUniv-w) + lower,
					Universe    +(kSizeUniv-w) + lower,
					sizeof(u_char)*width,
					cudaMemcpyHostToDevice,
					streams[stream]);
				}
				
				if(is_right_border())
				{
					cudaMemcpyAsync(UniverseGPU lower + w - width,
									Universe    lower + w - width,
									sizeof(u_char)*width,
									cudaMemcpyHostToDevice,
									streams[stream]);
				}
				if(is_low_right_border())
				{
					cudaMemcpyAsync(UniverseGPU lower     ,
									Universe    lower     ,
									sizeof(u_char)*width  ,
									cudaMemcpyHostToDevice,
									streams[stream]);
				}

				}*/
			
		
			// Emissao de Kernel <<<grid, block, Mem, #Stream>>>
			GOLInternalKernel<<<grids, blocks, 0, streams[stream]>>>
			(UniverseGPU + lower, NewUniverseGPU + lower);

			// Ultima Geracao Copiamos para o Host
			// if(Generation == NumberOfGenerations - 1)
			// 	cudaMemcpyAsync(Universe    + lower,   // ptr. para cpu
			// 					UniverseGPU + lower,   // ptr. para gpu
			// 					sizeof(u_char)*width,  // Tam. dos dados
			// 					cudaMemcpyDeviceToHost,// Sentido de copia 
			// 					streams[stream]);	   // qual fila de stream
			
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
	general_timer.stop("Avg. per Kernel", static_cast<double>(NumberOfGenerations));
	// Copiamos diretamente para o HOST
	general_timer.start();
	cudaMemcpy(Universe, UniverseGPU, kBinSize, cudaMemcpyDeviceToHost);
	general_timer.stop("Copy Memory D->H");
	// Necessario ?
	cudaDeviceSynchronize();
	comp_timer.stop("Total Time Execution");

	#ifdef  _DISPLAY_GAME
	DisplayGame(Universe);
	#endif/*_DISPLAY_GAME */
	
	
	// Destruimos os Streams
	for(int stream=0; stream < NumStreams; stream++)
		cudaStreamDestroy(streams[stream]);
	
	// Liberamos a memoria da GPU
	cudaFree(UniverseGPU);
	cudaFree(NewUniverseGPU);
}



#endif  /*_CUDA_KERNEL_H_*/