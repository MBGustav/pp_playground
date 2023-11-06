
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILENAME ("input.txt")
#define SEED (777)
#define MAX_ELEM (1<<20)

#define min(a,b) (((a)<(b))?(a):(b))

typedef struct input_data{
	int length;
	int data[MAX_ELEM];
}input_data;

input_data data;


void write_inputs(int nsamples)
{
	data.length = nsamples;
	FILE *f = fopen(FILENAME, "wb");

	for(int i = 0;i < nsamples; i++){
		data.data[i] = (int) rand();
	}
	
	if(fwrite(&data, sizeof(int), 1,f)<=0) printf("erro");
	if(fwrite(data.data, sizeof(int), nsamples, f) <= 0) printf("err") ;
	fclose(f);
}


int main(int argc, char **argv)
{

	srand(777);
	int N;
	int input = MAX_ELEM;
	if(argc != 2)
	{
		printf("Usage : ./input <Nr of values>\n");
		printf("Generating with:  %i\n", MAX_ELEM);	

	}
	if(argc == 2)
		input = atoi(argv[1]);

    N =  min(input,MAX_ELEM);
	write_inputs(N);

	return 0;
}