#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


int main(int argc, char **argv){

	FILE *fp=NULL;
	char fname[] = "input.txt";
	int N, S, seed = 777;
	if (argc <= 2) {
		printf ("ERROR: ./input-generator <size-vector> <size-subvector> <Optional:seed-random>\n");
		exit(1);
	}

	if(argc == 4){
		seed = atoi(argv[3]);
		printf("seed defined as - %d\n", seed);
		srand(seed);	
	} 


	N = atoi(argv[1]);
	S = atoi(argv[2]);
	fp = fopen(fname , "w" );

	if(N >= 8000000){
		printf ("ERROR: Size exceeded \n");
		return 1;
	}

	if(fp!= NULL){
		//insert total elements
		fprintf(fp, "N=%d\n", N);
		fprintf(fp, "S=%d\n", S);
		int rand_num;
		for(int i = 0; i < N; i++){
			rand_num = rand() % 100;
			fprintf(fp, "%d ",rand_num);
		}
		fclose(fp);
		printf("File created : %s\n", fname);
	}
	return(0);
}