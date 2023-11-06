#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define FILENAME ("input.txt")
#define SEED (777)
#define MAX_ELEM (1<<20)

typedef struct input_data{
	int length;
	int data[MAX_ELEM];
}input_data;
input_data data;

#define intSWAP(x,y) {int temp = x; x = y; y = temp;}

// Read Input from FILENAME
bool read_inputs()
{
	//read the amount of data
	FILE *f = fopen(FILENAME, "rb"); 
	if(!f) return false;
	if(fread(&data.length, sizeof(int), 1, f ) <= 0) return false;
	if(fread(data.data, sizeof(int), data.length, f) <= 0) return false;
	return true;
}

bool validate()
{
	for(int i = 0; i < data.length-1;i++)
	{
		if(data.data[i] > data.data[i+1])
			return false;
	}
	return true;
}


void oe_sorting()
{ 
	bool sorted = false;
	int *list = data.data;
	while(!sorted)
	{
		sorted = true;
		for (int i = 1; i < data.length - 1; i += 2) {
			if (list[i] > list[i + 1]) {
				intSWAP(list[i], list[i + 1]);
				sorted = false;
			}
		}

		for (int i = 0; i < data.length - 1; i += 2) {
			if (list[i] > list[i + 1]) {
				intSWAP(list[i], list[i + 1]);
				sorted = false;
			}
		}
	}
}



int main(){

	if(!read_inputs())
		printf("erro");
	
	printf("Sorting");
	oe_sorting(); 

	if(!validate()) printf("\nError ordenating!\n");
	else  printf("\n Correct!\n");


	return 0;
}