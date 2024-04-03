#ifndef _CUDAMATRIX_HPP__
#define _CUDAMATRIX_HPP__

#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_common.cuh"

//this is to reduce compiler c++ overhead
#ifndef data_t
#define data_t double
#endif /*data_t*/


#define MAX_DISPLAY_MATRIX (6)


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
    void display(int off_x = 0, int off_y = 0);

    // Returns row value
    int getRow() const;

    // Returns Col value 
    int getCol() const;
    
    // Returns Leading Dimension width
    int get_ld() const;
    
    // Total Amount of (real) data
    int getSize() const;
    
    // Total Amount of data used to alloc
    size_t MemSize( bool ) const;
    
    // device acess position
    data_t& at(int x, int y);
    data_t& const_at(int x, int y) const;        
    
    data_t* dataGPU();
    data_t*  data();
    // data_t*  data();
    
    void SynchronizeValues();
    
    // Transposed forced in memory
    void transposeInPlace(); 
    
    // Creates another matrix tranposing
    cudaMatrix transpose();
    
    // Notifies about changes in its values
    void changeOccurred(enum ChangeHandler);

    // Generate a random matrix On Host Only 
    void randMatrix(data_t lower_bound, data_t upper_bound, bool Synchronize = true);
    void Identity(data_t val = 1.0f, bool Synchronize = true);
    
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