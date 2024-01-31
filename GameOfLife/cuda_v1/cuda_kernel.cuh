#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_


#include "common_cuda.cuh"

__global__ void GOLInternalKernel(u_char *Universe, u_char *NewUniverse)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int gx = tx + blockDim.x * blockIdx.x;
	const int gy = ty + blockDim.y * blockIdx.y;

	// Out-of-bounds checking (we calc w/ wrap around)
	if (gx >= Width || gy >= Height)
		return;

	// Uso Memoria Local +2 para leitura das vizinhancas
	__shared__ u_char s_univ[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	// Condiciono as threads (0,x) e (y,0) para escrita nas bordas de s_univ

	if (tx == 0) {
		s_univ[ty + 1][0] = Universe[matrix_idx(gy + 1, gx)];
		s_univ[ty + 2][0] = Universe[matrix_idx(gy + 2, gx)];
	}

	if (ty == 0) {
		s_univ[0][tx + 1] = Universe[matrix_idx(gy, gx + 1)];
		s_univ[0][tx + 2] = Universe[matrix_idx(gy, gx + 2)];
	}

	// Topo superior
	if (ty == 0 && tx == 0) {
		s_univ[0][0] = Universe[matrix_idx(gy    , gx    )];
		s_univ[1][0] = Universe[matrix_idx(gy + 1, gx    )];
		s_univ[0][1] = Universe[matrix_idx(gy    , gx + 1)];
		s_univ[1][1] = Universe[matrix_idx(gy + 1, gx + 1)];
	}

	s_univ[ty + 2][tx + 2] = Universe[matrix_idx(gy + 2, gx + 2)];

	// Sincronização entre threads do grid
	__syncthreads();

	// Executa o calculo de posicionamento

	int n = s_univ[ty    ][tx] + s_univ[ty    ][tx + 1] + s_univ[ty    ][tx + 2] +
			s_univ[ty + 1][tx]                          + s_univ[ty + 1][tx + 2] +
			s_univ[ty + 2][tx] + s_univ[ty + 2][tx + 1] + s_univ[ty + 2][tx + 2];

	// New Universe is written
	NewUniverse[matrix_idx(gy + 1, gx + 1)] = ((n | s_univ[ty + 1][tx + 1]) == 3);
}


void GameOfLifeKernel(u_char *Universe, const int NumberOfGenerations)
{
    srand(777);
    //Game parameters
    constexpr int w= Width;
	constexpr int h= Height;
	const int kSizeUniv = Width * Height; 
	const int kBinSize = kSizeUniv * sizeof(u_char);
	int Generation =NumberOfGenerations;
    // GPU Parameters
    constexpr int nx_threads= THREADS_X;
	constexpr int ny_threads= THREADS_Y;
	u_char *UniverseGPU, *NewUniverseGPU;
	dim3 blocks(nx_threads, ny_threads);
	dim3 grids(_ceil(Width,nx_threads), _ceil(Height, ny_threads));

	const std::string label = "cuda_v1," +
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
	
	general_timer.start();
    while(Generation--){
        GOLInternalKernel<<<grids, blocks>>>(UniverseGPU, NewUniverseGPU);
    
        swap_pointers((void**)& UniverseGPU,(void**)& NewUniverseGPU);
        
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
    general_timer.stop("Copy Memory D->H");
	comp_timer.stop("Total Time Execution");
	#ifdef  _DISPLAY_GAME
	DisplayGame(Universe);
	#endif/*_DISPLAY_GAME */
	
	cudaFree(UniverseGPU);
	cudaFree(NewUniverseGPU);
}



#endif  /*_CUDA_KERNEL_H_*/