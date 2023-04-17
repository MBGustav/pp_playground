#ifndef __SERIAL_H__
#define __SERIAL_H__

#include <stdio.h> 

#include <stdlib.h> /*max()*/
#include <limits.h> /*INT_MIN*/
#ifndef max
	#define max(x,y) ((x)>(y) ? (x):(y))
#endif
int maxSum_serial(int *vec, int n, int s){

	//acumulamos a soma do vetor - faremos pela diferença
	int *aux = (int*) malloc((n+1) * sizeof(int));
	aux[0] = 0;
	//Acumulo a soma no vetor aux 
	for(int i = 1; i <=n;i++){
		aux[i] = aux[i-1]+vec[i-1];
	}
	int maxSum = INT_MIN;
	//Prefix Sum -- obtem o valor da soma por meio da diferença
	for(int i = 0; i <=n; i++){
		int sum = aux[i] - aux[i-s];
		maxSum = max(maxSum, sum);
	}

	free(aux);
	return maxSum;
}


#endif /*__SERIAL_H__*/