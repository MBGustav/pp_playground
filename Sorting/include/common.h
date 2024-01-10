#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


#define SWAP(x,y) {int temp = x; x = y; y = temp;}
#define intSWAP(x,y) {int temp = x; x = y; y = temp;}
#define min(a,b) (((a)<(b))?(a):(b))


#define WIDTH_REPRESENTATION (5)
#define MAX_ELEM (1<<20)
#define FILENAME ("bin/input.in")
#define SEED (777)
#define N_THREADS (1024)


typedef struct input_data{int length, data[MAX_ELEM];}input_data;
input_data data;


bool validate()
{
	for(int i = 0; i < data.length-1;i++)
	{
		if(data.data[i] > data.data[i+1])
			return false;
	}
	return true;
}



void print_values()
{
	printf("Displaying vector\n");
	if(min(data.length,WIDTH_REPRESENTATION) != data.length)
		printf("Showing the firsts %d elements\n", WIDTH_REPRESENTATION);

	for(int i=0; i < min(data.length,WIDTH_REPRESENTATION); i++)
	{
		printf("%5d ", data.data[i]);
	}
    printf("\n");
}


void error_input()
{
	printf("Error - input  file not found!\n");
	exit(EXIT_FAILURE);
}
// Read Input from FILENAME
bool read_inputs()
{
	//read the amount of data
	FILE *f = fopen(FILENAME, "rb");
	if(!f) 
	{
		printf("Not Found");
		return false;
	}
	if(fread(&data.length, sizeof(int), 1, f ) <= 0) return false;
	if(fread(data.data, sizeof(int), data.length, f) <= 0) return false;
	return true;
}







#endif /*COMMON_H*/