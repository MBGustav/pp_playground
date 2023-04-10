#include <iostream>

#include "include/PNM.h"
#include "include/structs_inputs.h"
using namespace std;



int main(int argc, char **argv) {
    
    int IMG_SIZE = atoi(argv[1]);
    
    //Define Tamanho de vetor (r,g,b)
    int size = IMG_SIZE * IMG_SIZE * 3;
    printf("Execucao openMP:\n\tsize=%d\n", IMG_SIZE);

    unsigned char * pixels = new unsigned char[size];

    JuliaFilter(pixels, IMG_SIZE);

    write_pnm("julia_omp.pgm", pixels, IMG_SIZE);

    free(pixels);
    return 0;
}

