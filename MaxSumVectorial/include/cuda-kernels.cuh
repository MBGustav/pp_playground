#ifndef __CUDA_KERNEL_CUH__
#define __CUDA_KERNEL_CUH__

#define ThreadsPerBlock (64)

//devo usar esta implementação ?
// __device__ int max(int x, int y){return x > y ? x : y;}

__global__ void sumMaxVector(int *vector, int n,int len, int *partial_sum){

    int tx = threadIdx.x + (blockIdx.x * blockDim.x); 
    if(tx < n){   

        __shared__ int cache[ThreadsPerBlock];
        int cacheIdx = threadIdx.x;

        int sum=0;
        for(int i = 0; i < len; i++){
            int offset = tx + i;
            sum += vector[offset];
        }

        cache[cacheIdx] = sum;

        __syncthreads();

        int i = blockDim.x/2;
        while(i !=0){
            if(cacheIdx < i ){
                int v2 = cache[cacheIdx+i];
                int old = atomicMax(&cache[cacheIdx],v2);
                }

            // if(tx == 0){
            //     for(int i = 0; i < ThreadsPerBlock; i++)
            //         printf("%d ",cache[i]);
            //     printf("\n");
            // }
                i/=2;
                __syncthreads();
            }

            if(cacheIdx==0)
                partial_sum[blockIdx.x] = cache[0];
        }
    }
int maxSum_cuda(int *vector, int n, int len){

    //Variaveis internas
    int max_sum = 0;
    // Size: n-len
    int blocksPerGrid = min(32, (n+ThreadsPerBlock-1-len)/ThreadsPerBlock);
    int *d_vector, *d_partial_sum, *partial_sum;
    int sizeof_vector = n * sizeof(int);
    int sizeof_partial= blocksPerGrid * sizeof(int); 

    //Aloca memoria - host e device
    cudaMalloc(&d_vector, sizeof_vector);
    partial_sum = (int*) malloc(sizeof_partial);
    cudaMalloc(&d_partial_sum, sizeof_partial);

    //Define valores iniciais dos vetores no device
    cudaMemcpy(d_vector, vector, sizeof_vector, cudaMemcpyHostToDevice);
    // cudaMemset(d_partial_sum, 0x0, sizeof_partial);

    //Calcula soma vetorial
    sumMaxVector<<<blocksPerGrid, ThreadsPerBlock>>>(d_vector, n, len, d_partial_sum);

    //Retorna do device - maximos parciais do vetor
    cudaMemcpy(partial_sum, d_partial_sum, sizeof_partial, cudaMemcpyDeviceToHost);
    
    //Busco resultados via CPU: iteração mais eficiente 
    //TODO: comparar resultado associado a openMP aqui
    for(int i = 0; i< blocksPerGrid; i++){
        max_sum = max(max_sum, partial_sum[i]);
    }

    free(partial_sum);
    cudaFree(d_partial_sum);
    cudaFree(d_vector);


    return max_sum;
}








#endif /*__CUDA_KERNEL_CUH__*/