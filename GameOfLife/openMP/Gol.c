#include <stdio.h>

#include "common.h"

const int w = Width;
const int h = Height;
u_char univ[Width][Height];


int main(int c, char **v)
{
	int g = 1;
	
	if(c == 1) 
		DisplayBanner();
	
	if (c > 1) g = MAX(g, atoi(v[1]));
	game(univ, g);
}