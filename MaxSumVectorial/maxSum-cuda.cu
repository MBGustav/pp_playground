#include <stdio.h>
#include <stdlib.h>

#include "include/cuda-kernels.cuh"


int main(void){
    int N, S, maxSum;
    scanf("N=%d\n",&N);
    scanf("S=%d\n",&S);
    
    int *vector = (int*)malloc(N*sizeof(int)); 
    
    for(int i = 0; i < N; i++)
        scanf("%d ",&(vector[i]));
    

    maxSum = maxSum_cuda(vector, N, S);
    
    printf("Resultado: %d\n", maxSum);

    return 0;
}

