
#include <stdio.h>
#include <iostream>

#include "common_cuda.cuh"
#include "cuda_kernel.cuh"

constexpr int w = Width;
constexpr int h = Height;
unsigned char univ[w*h]; 



int main(int c, char **v)
{
	int g = 1;
	
	if(c == 1) 
		DisplayBanner();
	

 	if (c > 1) g = MAX(g, atoi(v[1]));


	GameOfLifeKernel(univ, g);
}