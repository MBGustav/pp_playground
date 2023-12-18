// ======================= CUDA_MATRIX_CPP ======================

#include <stdio.h>
#include <stdlib.h>

#include "cudaMatrix.h"

int cudaMatrix::size() const {return _row * _col;}

// Returns row value
int cudaMatrix::getRow() const { return this->_row;}

// Returns Col value 
int cudaMatrix::getCol() const { return this->_col;}

// Returns Leading Dimension width
int cudaMatrix::getLd() const { return this->_ld;}

int cudaMatrix::MemSize(bool bin=false) const{
    return getLd() * getRow() * (bin ? sizeof(data_t) : 1 );
}

void cudaMatrix::transpose()
{
    is_transposed = !is_transposed;
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
    is_transposed=false;
    Change = Equal;
}
void cudaMatrix::deallocate()
{
    free(host_data);
    cudaFree(dev_data);
    check_last_error ();
}

// bool cudaMatrix::reshape(int new_row, int new_col)
// {
//     if( this->size() != new_col * new_row )
//         return false;

// }


cudaMatrix::cudaMatrix() : host_data{NULL}, dev_data{NULL}{}

cudaMatrix::cudaMatrix(int row, int col) : _row(row), _col(col){allocate(row, col);}

cudaMatrix::~cudaMatrix() {deallocate();}



void cudaMatrix::display() const
{
    if(!host_data){
        fprintf(stderr, "Invalid Acess Memory - from host.\n");
        return;
    }

    int scale_r = std::min(MINIMUM_DISPLAY,_row);
    int scale_c = std::min(MINIMUM_DISPLAY,_col);
    printf("\nDisplaying Matrix");
    printf(scale_r == MINIMUM_DISPLAY || scale_c == MINIMUM_DISPLAY? " showing top left:\n": ":\n");
    for(int ii=0; ii< scale_c; ii++){
    for(int jj=0; jj< scale_c; jj++){
        printf("% 15.3f  ", host_data[is_transposed ? 
                idx_matrix(ii,jj,_ld):idx_matrix_transp(ii,jj,_ld)]);
    }
    printf("\n");
    }

}

//param: out(bool) check if offload executed correctly
bool cudaMatrix::randMatrixOnHost(bool Synchronize = true)
{
    bool ret = false;

    int ld = getLd();
    for(int ii=0; ii< getRow(); ii++)
    for(int jj=0; jj< getCol(); jj++)
        this->host_data[idx_matrix(ii,jj,ld)] = rand();


    if(Synchronize){
        ret = this->TransferData(cudaMemcpyHostToDevice);
        Change = ret ? Equal : ChangeOnHost;
        return ret;
    } // else do Not sync


    return true;
}

bool cudaMatrix::TransferData(cudaMemcpyKind kind)
{
    cudaMemcpy(dev_data, host_data, sizeof(data_t) * MemSize(), kind);
    check_last_error();
    Change = Equal;
    return true;
}

