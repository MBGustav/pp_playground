#include<stdio.h>
#include<omp.h>
#include<assert.h>


const int N = 20;
int main()
{
  float A[N]; 
  float B[N]; 
  // Fill vectors
  for (int i = 0; i < N ; i++) {
    A[i] = 1.5;
    B[i] = 10.5; 
  }

  for(int i = 0; i < N; i++)
    A[i] = A[i] + B[i];

  return 0;
}
