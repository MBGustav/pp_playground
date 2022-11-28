#include<stdio.h>
#include<omp.h>

int main(){

  #pragma omp parallel
  { // Inicio Paralelização
    int ID = omp_get_thread_num();
    printf("Hello thread %d  ", ID);
    printf("Same thread %d\n", ID);

  } // Final paralelização
}
