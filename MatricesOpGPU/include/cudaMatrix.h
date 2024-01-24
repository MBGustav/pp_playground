#ifndef _CUDAMATRIX_HPP__
#define _CUDAMATRIX_HPP__

#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_common.h"

//this is to reduce compiler c++ overhead
#ifndef data_t
#define data_t double
#endif /*data_t*/


#define MAX_DISPLAY_MATRIX (6)
/*Enum to deal with modifications between Host <-> Device*/


// template<typename TT>
class cudaMatrix
{

private:
    // Matrix Parameters:
    int _row, _col,_ld;
    bool is_transposed;


    ChangeHandler Change;


    // Two pointers: local and device
    data_t *host_data, *dev_data;
    void allocate(int row, int col);
    void deallocate();

    //This Method deals with synchronization(H<->D)
    bool TransferData(cudaMemcpyKind kind);

public:
    // Show top left matrix

    void display() const;

    // Returns row value
    __host__ __device__ int getRow() const;

    // Returns Col value 
    __host__ __device__ int getCol() const;
    
    // Returns Leading Dimension width
    __host__ __device__ int getLd() const;

    // Total Amount of (real) data
    __host__ __device__ int getSize() const;

    __host__ size_t MemSize( bool ) const;

    // device acess position
    __host__ __device__ data_t& at(int x, int y);
    __device__ __host__ data_t& const_at(int x, int y) const;        

    __host__ data_t& getDataGpu();

    __host__ data_t&  data();
    
    void SynchronizeValues();

    // Generate a random matrix On Host Only 
    void randMatrix(data_t lower_bound, data_t upper_bound, bool Synchronize);

    // Transposed forced in memory
    void transposeInPlace(); 

    // Creates another matrix tranposing
    cudaMatrix transpose();


    // Matrix Constructors and Destructors
    cudaMatrix(int row, int col);
    cudaMatrix();
    ~cudaMatrix();
};


class GPU_properties{
    // TODO: include multiples GPUS
};

class Event_Handler
{
    // TODO: How to sync ?? 
    // Probably an output from each Matricial Operation
};








#endif /*_CUDAMATRIX_HPP__*/