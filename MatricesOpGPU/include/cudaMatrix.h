#ifndef _CUDAMATRIX_HPP__
#define _CUDAMATRIX_HPP__

#include <cuda_runtime.h>
#include "cuda_common.h"

//this is to reduce compiler c++ overhead
#ifndef data_t
#define data_t double
#endif /*data_t*/

/*Enum to deal with modifications between Host <-> Device*/

class cudaMatrix
{
private:

    // auto GPUDevice /*Which GPU is located*/
    int _row, _col,_ld;
    bool is_transposed;

    ChangeHandler Change;


    // Two pointers: local and device
    data_t *host_data, *dev_data;
    void allocate(int row,int col);
    void deallocate();

    //This Method deals with synchronization(H<->D)
    bool TransferData(cudaMemcpyKind kind);

public:
    int MemSize(bool bin) const;
    // Show top left matrix
    void display() const;

    //Return matrix size
    int size() const;

    // Returns row value
    int getRow() const;

    // Returns Col value 
    int getCol() const;
    
    // Returns Leading Dimension width
    int getLd() const;

    void transpose();

    // Generate a random matrix On Host Only 
    bool randMatrixOnHost(bool Synchronize);

    __device__ data_t* getDataGpu();

    // Generate a random matrix On Device Only  -> TODO
    // bool randMatrixOnDevice(bool DevSynchronize = true);

    // Matrix Constructors and Destructors
    cudaMatrix(int row, int col);
    cudaMatrix();
    ~cudaMatrix();
};


class GPU_properties{
    // for future: include multiples GPUS
};

class Event_Handler
{
    // TODO: How to sync ?? 
    // Probably an output from each Matricial Operation
};



// ======================= CUDA_MATRIX_CPP ======================

// #include <stdio.h>
// #include <stdlib.h>

// #include "cudaMatrix.h"

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


// C = alpha x A B + beta C
void Multiply(double alpha, cudaMatrix &A, cudaMatrix &B,double beta, cudaMatrix &C, bool sync_devices)
{

    OffloadSelect Ofs = DeviceSwitch(A,B,C);
    switch(Ofs)
    {
    case GPU_01: Wrapper_matmul01(A,B,C, alpha, beta);
        break;
    case GPU_02: Wrapper_matmul02(A,B,C, alpha, beta);
        break;
    case CPU: Matmul(A,B,C, alpha, beta);
        break;

    default: printf("[ERROR] Device Selection error\n");
    }

}


__device__ data_t* cudaMatrix:: getDataGpu(){return dev_data;}







#endif /*_CUDAMATRIX_HPP__*/