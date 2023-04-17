#include <stdio.h>
#include <stdlib.h>

#include "include/serial.h"


int main(void){
    int N, S, maxSum, *vector =NULL;
    if(scanf("N=%d\n",&N)!=1) return EXIT_FAILURE;
    if(scanf("S=%d\n",&S)!=1) return EXIT_FAILURE;
    vector = (int*)malloc(N*sizeof(int)); 
    
    if(vector == NULL)  return EXIT_FAILURE;
    
    for(int i = 0; i < N; i++)
        if(scanf("%d ",&(vector[i])) !=1) return EXIT_FAILURE;
    

    maxSum = maxSum_serial(vector, N, S);
    
    printf("Resultado: %d\n", maxSum);

    return 0;
}

