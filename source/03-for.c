#include<stdio.h>
#include<omp.h>
#include<assert.h>

void sum(float val, float *A, int N)

const int N = 20;
int main(){

  float A[N]; 
  float B[N]; 
  // Fill vectors
  for (int i = 0; i < N ; i++) {
    A[i] = 1.5;
    B[i] = 10.5; 
    }

  #pragma omp parallel num_threads(4)
  { //Abre um pool para "x" threads, e dividimos o trabalho entre si
    int id, i, n_threads, i_start, i_end; 
    id = omp_get_thread_num();
    n_threads = omp_get_num_threads(); 
    
    //divisao de elementos pelo numero de threads
    i_start =  id    * N / n_threads ; 
    i_start = (id+1) * N / n_threads ;

    //impede de acessar memoria indevida
    if(id == n_threads -1) iend = N; 
    
    #pragma omp for
    for( i=i_start; i < iend; i++){
      A[i] = A[i] + B[i];
    }     

  } 

  return 0;
}
