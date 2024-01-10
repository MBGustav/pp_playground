#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "common.h"

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
	
	if(!read_inputs()) error_input();
	
	printf("Total elements %i\n", data.length);
	oe_sorting(); 


	if(!validate()) printf("\nError ordenating!\n");
	else  printf("\n Correct!\n");

	print_values();


	return 0;
}