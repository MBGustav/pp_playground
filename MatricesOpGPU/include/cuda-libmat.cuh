
#include "cuda_common.hpp"

struct cuda_matrix
{
    cuda_matrix(sycl::queue &main_queue) : q{main_queue}, data{NULL} {};
    void deallocate(){if (data) free(data, q.get_context());}

    data_t *data, *d_data;
    int row, col, ldw;
    int size;
    
    cuda_matrix() : data{NULL},  d_data{NULL} {};

    void allocate(int _row, int _col, bool alloc_gpu=true){
        if(_row <1 || _col<1) {
            fprintf(stderr, "%s:%d: memory allocation error\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        row = _row;
        col = _col;
        ldw = getld(col); // leading dimension
        size = ldw*row;

       	data = (data_t*) malloc( size * sizeof(d_type));
        if(alloc_gpu) cudaMalloc(&d_data, size * sizeof(d_type));
    }

    void rnd_matrix(bool rand = true){
        for(int i=0; i < row; i++)
            #pragma simd vectorlength(8)
            for(int j=0; j < col; j++)
                data[matrix_idx(i, j, ldw)] = (rand ? (data_t) (std::rand() %100) : i*col + j);
    }

    void DataToGPU()
    {

    }
    
    void printMat(){
        int i, j;
        printf("Printing top left:\n");
        for (i = 0; i < std::min(6, row); i++) {
            printf("\n");
            for (j = 0; j < std::min(6, col); j++)
                printf("  %010.4lf", data[matrix_idx(i, j, ldw)]);
        }
        printf("\n\n");
        
    }

    data_t* data_NoPadding()
    {
        data_t *data_nopad = (data_t*) malloc(row * col * sizeof(data_t));
        //copy data changing leading dimension
        for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            data_nopad[matrix_idx(i,j,col)] = data[matrix_idx(i,j,ldw)];
            
        return data_nopad;
    }


    // ~cuda_matrix() { deallocate(); }
};


__global__ void kernel_matmul(d_type *A, d_type *B, d_type *C,int m, int k, int n)
{

	int tx = threadIdx.x;
	int ty = threadIdx.y; 

	int row = blockIdx.x * blockDim.x + tx;
	int col = blockIdx.y * blockDim.y + ty;

	d_type tmp=0;
	if(m < row && n < col){
		for(int i=0; i < k; i++)
			tmp += A[row*m+i]*B[i*k+col];
		// __syncthreads();
		C[row*m+col] = tmp;
	}
	if(threadIdx.x == 0)
		printf("[%d,%d] = %f\n",row,col, tmp);
}

void matmul_wrapper(matrix *A, matrix *B, matrix *C){

	d_type *d_A, *d_B, *d_C;
	int m = A->row;
	int k = A->col;
	int n = B->col;	
	int sizeA = m*k;
	int sizeB = k*n;
	int sizeC = m*n;

	dim3 NumThreads(NTHREADS, NTHREADS);
    dim3 BlkPerGrid((n + NumThreads.x - 1) / NumThreads.x, (m + NumThreads.y - 1) / NumThreads.y);


 	
 	cudaMalloc(&d_A, sizeA * sizeof(d_type));
 	cudaMalloc(&d_B, sizeB * sizeof(d_type));
 	cudaMalloc(&d_C, sizeC * sizeof(d_type));

 	cudaMemcpy(d_A, A->elem, sizeA*sizeof(d_type), cudaMemcpyHostToDevice );
 	cudaMemcpy(d_B, B->elem, sizeB*sizeof(d_type), cudaMemcpyHostToDevice );

 	kernel_matmul<<<BlkPerGrid, NumThreads>>>(d_A, d_B, d_C, m, k, n);
 	cudaDeviceSynchronize();

	cudaMemcpy(C->elem, d_C, sizeC*sizeof(d_type), cudaMemcpyDeviceToHost);
}
