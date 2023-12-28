#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <stdio.h>

// Not-templated: not mixing datatypes
#ifndef data_t
#define data_t double
#endif

typedef struct Matrix
{
    int Row,Col;
    data_t *data;

    Matrix(){data = NULL;}    
    Matrix(int kRow, int kCol): Row(kRow), Col(kCol){allocate(Row, Col);}
    
    void allocate(int r,int c){data = (data_t*)malloc(sizeof(data_t) * r*c);}
    
    int size(){return Row*Col;}
    

}Matrix;



#endif //_MATRIX_HPP_