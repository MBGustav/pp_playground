#include<stdio.h>
#include<omp.h>
#include <sys/time.h>
const int N = 1000;
int main()
{
  int A[N];
  int B[N];
  struct timeval start;
  struct timeval end;
  // Fill vectors
  
  for (int i = 0; i < N ; i++) {
    A[i] = 1;
    B[i] = 10; 
  }

  gettimeofday(&start, NULL);
  //start = time(NULL); //omp_get_wtime();
  for(int i = 0; i < N; i++)
    A[i] = A[i] + B[i];
  gettimeofday(&end, NULL);
  //end = time(NULL); // omp_get_wtime();
  double t_time = end.tv_sec*1000000 + end.tv_usec - start.tv_sec*1000000 + start.tv_usec;
  printf("Serial Time: %lf ms\n", t_time*1e-6);
  return 0;
}
