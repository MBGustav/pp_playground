#include<stdio.h>
#include<omp.h>
#include<assert.h>

void fill(int pos, float val, float* A, int N)
{
  A[pos] = val;
  printf("A[%d] = %.4f\n", pos, val);
}

const int N = 1000;
int main(){

  float A[N]; 
  #pragma omp parallel num_threads(4)
  { 
    int ID = omp_get_thread_num();
    fill(ID, 10.8, A, N);

    

  } 
}
