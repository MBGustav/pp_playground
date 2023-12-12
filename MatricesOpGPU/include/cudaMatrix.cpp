
#include <stdio.h>
#include <stdlib.h>

#include "cudaMatrix.h"


cudaMatrix::cudaMatrix() : host_data{NULL}, dev_data{NULL}
{
}
cudaMatrix::cudaMatrix(int row, int col) : _row(row), _col(col)
{
    allocate(row, col);
}


cudaMatrix::~cudaMatrix()
{
}

void cudaMatrix::allocate(int row,int col)
{
    this->_row = row;
    this->_col = col;
    // to make padding efficient
    this->_ld  = getld(col);

    // TODO: should I implement here te Async Memory, or alloc through the op? No..
    
    MALLOC(host_data, this->MemSize(true));
    cudaMalloc(&dev_data, this->MemSize(true));
    check_last_error();

}
void cudaMatrix::deallocate(int r,int c)
{
    cudaFree(dev_data);
    check_last_error ();
}


int cudaMatrix::MemSize(bool bin=false)
{
    int size =  _row * _ld;
    return bin * size + !(bin) * size ;
}
int cudaMatrix::size(bool bin=false)
{
    int size =  _row * _col;
    return bin * size + !(bin) * size ;
}