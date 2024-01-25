#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdlib.h>
#include <stdio.h> 
#include <unistd.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define for_x for (int x = 0; x < w; x++)
#define for_y for (int y = 0; y < h; y++)
#define for_xy for_x for_y
#define for_yx for_y for_x


#ifndef Height
#define Height (1<< 10)
#endif

#ifndef Width
#define Width  (1<< 10)
#endif

#ifndef THREADS_Y
#define THREADS_Y (32)
#endif

#ifndef THREADS_X
#define THREADS_X (32)
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif 
typedef unsigned char u_char;

// global parameter
void swap_pointers(void **ptrA, void **ptrB)
{
	void *temp = *ptrA;
	*ptrA = *ptrB;
	*ptrB = temp;
}


// > Linearize to 1D Matrix
#define idx_matrix1D(x, y) \
    (((x + Width) % Width) + ((y + Height) % Height) * Width)


void show2D(void *univ, int w, int h)
{	
	u_char (*u)[Width] = (u_char (*)[Width]) univ;
	printf("\033[H");
	for_y {
		for_x printf(u[y][x] ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
	fflush(stdout);
}

void show1D(u_char *univ, int w, int h)
{

	printf("\033[H");
	for_y {
		for_x printf(univ[idx_matrix1D(y, x)] ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
	fflush(stdout);
}



void evolve_serial2D(void *universe, int w, int h)
{	
	u_char new_univ[Height][Width];
	u_char (*univ)[Width] = (u_char (*)[Width]) universe;

	for_yx {
		u_char n = univ[(y-1+h)%h][(x-1+w)%w] + univ[(y-1+h)%h][(x  +w)%w] + univ[(y-1+h)%h][(x+1+w)%w] +
		           univ[(y  +h)%h][(x-1+w)%w] +                              univ[(y  +h)%h][(x+1+w)%w] +
		           univ[(y+1+h)%h][(x-1+w)%w] + univ[(y+1+h)%h][(x  +w)%w] + univ[(y+1+h)%h][(x+1+w)%w]; 
		new_univ[y][x] = ((n | univ[y][x]) == 3);
	}
	for_yx univ[y][x] = new_univ[y][x];
	// swap_pointers((void**)& univ,(void**)& new_univ);
}


void game(void *univ, int NumGenerations)
{
	srand(777);
	const int h = Height;
	const int w = Width;

	u_char (*univ2D)[Width] = (u_char (*)[Width]) univ;


	for_xy univ2D[y][x] = rand() < RAND_MAX / 10 ? 1 : 0;
	// for_xy univ2D[y][(x*w)%h] = y == x ? 1 : 0;
	while (NumGenerations-- )
	{
		evolve_serial2D(univ, w, h);
		#ifdef _DEBUG_PER_STEP
		show2D(univ2D, w, h);
		getchar();
		#endif /*_DEBUG_PER_STEP*/
	}

	show2D(univ2D, w, h);
}



void DisplayBanner()
{
    printf("   Game of Life Properties:      \n");
    printf("\t Usage: ./gol NumGen \n");
    // printf("\t in: Filename to be read   \n");
    // printf("\t out: Filename to be written (std = cout) \n");
	printf("\t NumGen: Number of Generations expected\n");
	exit(EXIT_FAILURE);
}



#endif /*_COMMON_H_*/