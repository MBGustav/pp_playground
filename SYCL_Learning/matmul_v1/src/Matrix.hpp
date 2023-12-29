#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <stdio.h>

// Not-templated: not mixing datatypes
#ifndef data_t
#define data_t double
#endif

#define MINIMUM_DISPLAY (8)


typedef struct Matrix
{
    int Row, Col;
    data_t *data;

    Matrix(){data = NULL;}    
    Matrix(int kRow, int kCol): Row(kRow), Col(kCol){allocate(Row, Col);}
    
    void allocate(int r,int c){data = (data_t*)malloc(sizeof(data_t) * r*c);}
    
    int size(){return Row*Col;}
    

}Matrix;


void DisplayMatrix(Matrix &A)
{
    std::cout << "Displaying Matrix \n"<<std::endl;
    for (int row = 0; row < std::min(MINIMUM_DISPLAY, A.Row); row++) {
        for (int col = 0; col < std::min(MINIMUM_DISPLAY, A.Col); col++) {
            // Edit the output format of cout
            std::cout << std::fixed << std::setprecision(2);

            // Print the results
            std::cout << std::setw(8) << A.data[col * A.Row + row] << " ";
        }
        std::cout << std::endl;
    }
}

#endif //_MATRIX_HPP_