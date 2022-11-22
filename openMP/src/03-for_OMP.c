#include<stdio.h>
#include<omp.h>
#include<time.h>
#include<assert.h>


const int N = 1000;
int main(){
  int A[N]; 
  int B[N]; 
  // Fill vectors
  for (int i = 0; i < N ; i++) 
  {
    A[i] = 1;
    B[i] = 10; 
  }
  
  double start = omp_get_wtime();
  /*Abre um pool para "x" threads, e dividimos o trabalho entre si
    Nesse caso, teremos 4 "caminhos" que executam uma parte da soma vetorial
  */
  #pragma omp parallel
  {
    int id, i, n_threads, i_start, i_end; 
    id = omp_get_thread_num();
    n_threads = omp_get_num_threads(); 
    
    //divisao de elementos pelo numero de threads
    // i_start =  id    * N / n_threads ; 
    //i_end   = (id+1) * N / n_threads ;

    //impede de acessar memoria indevida
    //if(id == n_threads -1) i_end = N;
    #pragma omp parallel for
    for( i=0; i < N; i++){
      A[i] = A[i] + B[i];
    }     
  }

  double end = omp_get_wtime();
  printf("Parallel Time: %.4lf ms\n", (end-start)*1e3);
  return 0;
}
