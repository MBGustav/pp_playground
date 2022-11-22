#include<stdio.h>
#include<omp.h>
#include<time.h>

#define N 500


void matrix_mul(int A[N][N], int B[N][N], int C[N][N])
{
  int i, j, k;
  for ( i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      for (k = 0; k < N; k++)
        C[i][j] += A[i][k] * B[k][j]; 
    }
  }
}

void matrix_mul_OMP(int A[N][N], int B[N][N], int C[N][N])
{
  int i, j, k;
  #pragma omp parallel for private(i, j, k) shared(A,B,C)
  for ( i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      for (k = 0; k < N; k++)
        C[i][j] += A[i][k] * B[k][j]; 
    }
  }

}

int main(){
  
  double start_s, end_s, start_p, end_p;
  int i, j;
  int A[N][N]; 
  int B[N][N]; 
  int C_OMP[N][N];
  int C_serial[N][N];
  // Fill vectors
  for (i = 0; i < N ; i++) {
    for (j = 0; j < N ; j++) {
      A[i][j] = i*i;
      B[i][j] = -i; 
    }
  }
  
  start_s = omp_get_wtime();
  matrix_mul(A, B, C_serial);
  end_s = omp_get_wtime();
  
  
  printf("Serial Time:   %.4lf ms\n", end_s-start_s);

  start_p = omp_get_wtime();
  matrix_mul_OMP(A, B, C_OMP);
  end_p = omp_get_wtime();
  
  printf("Parallel Time: %.4lf ms\n", end_p-start_p);
  printf("Speed-up :     %.4lf ", (end_s-start_s)/(end_p-start_p));
  return 0;
  
}

