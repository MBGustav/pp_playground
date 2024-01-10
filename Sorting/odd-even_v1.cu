
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "helper.cuh"

#include "common.h"



__global__ void kernel_oesort(int *data, int n)
{
	int tx = threadIdx.x;
	int gx =  blockDim.x * blockIdx.x + tx;

	 if (gx < n - 1) { //bound check
        if (gx % 2 == 0) //odd 
            if (data[gx] > data[gx + 1])
                SWAP(data[gx], data[gx + 1]);
    
        if(gx % 2 != 0) //even
            if (data[gx] > data[gx + 1])
                SWAP(data[gx], data[gx+1]);
		
		__syncthreads();
    }
}


void wrapper_sorting()
{
	// Set parameters to GPU
	int *dev_data;
	int *host_data = data.data;
	int  kDataSizeBytes = data.length * sizeof(int);
	int NumBlocks = sdiv(data.length, N_THREADS);
	int stream;
	uint64_t lower, upper, width;
	const int stream_chunk_size = N_THREADS; //  A len and a half 
	int stride = (N_THREADS+1) / 2;

	
	// Alloc Space in GPU
	cudaMalloc(&dev_data, kDataSizeBytes);
	check_last_error();
	
	cudaMemcpy(dev_data, data.data, kDataSizeBytes, cudaMemcpyHostToDevice);
	check_last_error();

	// Set Streams parameters to GPU
	// The total streams depends on total data being sorted
	int num_streams = 10;
	cudaStream_t streams[num_streams];	
	
	// Creating the total Streams and Storing
	for (stream = 0; stream < num_streams; stream++)
		cudaStreamCreate(&streams[stream]);

	/*
	We can rearrange the total elements being sorted
	in accord to the total threads available. Although, 
	is necessary re-sort some areas intersecting each other.
	See more on README.
	*/
	
	for(int i = 0; i < data.length ; i++){
	// Send Concurrent kernel (odd step)
		for(stream=0; stream < num_streams; stream++)
		{
			lower = stream_chunk_size*stream;				   // Pointer to First element
			upper = min(lower+stream_chunk_size, data.length); // Pointer to Last element
			width = upper-lower;							   // Total Elements
			//How to divide btwn them ?
			// printf("lower = %d\n",(int) lower);
			
			kernel_oesort<<<1 ,N_THREADS,0,streams[stream]>>>(dev_data+lower, width);		
		}
		// Sync before even step
	// Send Concurrent kernel (even step)
		for(stream=0; stream < num_streams; stream++)
		{
			lower = stride + stream_chunk_size*stream;		   // Pointer to First element
			upper = min(lower+stream_chunk_size, data.length - stride); // Pointer to Last element
			width = upper-lower;							   // Total Elements
			//How to divide btwn them ?
			kernel_oesort<<<1 ,N_THREADS,0,streams[stream]>>>(dev_data+lower, width);		
		}
	}
		
	// Synch all streams before copy back
	for(stream=0; stream < num_streams; stream++)
		cudaStreamSynchronize(streams[stream]);
		
	// Copy back to Host all sorted data
	cudaMemcpy(host_data, dev_data,
			kDataSizeBytes, cudaMemcpyDeviceToHost);

	// Destroy Streams Created
	for (stream = 0; stream < num_streams; stream++) {
		cudaStreamDestroy(streams[stream]);
	}



}



int main(){

	if(!read_inputs()) error_input();

	printf("Total elements %i\n", data.length);
	// print_values();
	wrapper_sorting(); 
	cudaDeviceSynchronize();

	

	if(!validate()) printf("\nError ordenating!\n");
	else  printf("\n Correct!\n");

	print_values();

	return 0;
}
