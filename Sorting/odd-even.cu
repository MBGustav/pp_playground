
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "helper.cuh"
#define FILENAME ("input.txt")
#define SEED (777)
#define MAX_ELEM (1<<20)
#define N_THREADS (124)


#define SWAP(x,y) {int temp = x; x = y; y = temp;}
#define min(a,b) (((a)<(b))?(a):(b))

typedef struct input_data{int length, data[MAX_ELEM];}input_data;
input_data data;

// Read Input from FILENAME
bool read_inputs()
{
	//read the amount of data
	FILE *f = fopen(FILENAME, "rb"); 
	if(!f) return false;
	if(fread(&data.length, sizeof(int), 1, f ) <= 0) return false;
	if(fread(data.data, sizeof(int), data.length, f) <= 0) return false;
	

	return true;
}

bool validate()
{
	for(int i = 0; i < data.length-1;i++)
	{
		if(data.data[i] > data.data[i+1])
			return false;
	}

	return true;
}


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
	//alloc space in GPU and offload 
	int *dev_data;
	int *host_data = data.data;
	int  numBytes  = data.length * sizeof(int);
	int NumBlocks = sdiv(data.length, N_THREADS);
	Timer t(0);
	t.start();
	cudaMalloc(&(dev_data), numBytes);
	check_last_error();
	cudaMemcpy(dev_data, host_data, numBytes, cudaMemcpyHostToDevice);
	check_last_error();
	t.stop("Data Offload H -> D");
	//send data to 
	for(int i=0; i < data.length; i++){
		kernel_oesort<<< NumBlocks, N_THREADS>>>(dev_data, data.length);
	}
	t.stop("Kernel calculation");
	check_last_error();
	// data back to gpu and its over :D
	cudaMemcpy(host_data, dev_data, numBytes , cudaMemcpyDeviceToHost);
	check_last_error();
	cudaDeviceSynchronize();
}


int main(){

	if(!read_inputs())
		printf("error\n");

	printf("Total elements %i\n", data.length);
	wrapper_sorting(); 


	if(!validate()) printf("\nError ordenating!\n");
	else  printf("\n Correct!\n");


	return 0;
}
