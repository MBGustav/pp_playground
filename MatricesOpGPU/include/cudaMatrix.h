#ifndef _CUDAMATRIX_HPP__
#define _CUDAMATRIX_HPP__

#include "cuda_common.h"

//this is to reduce compiler c++ overhead
#ifndef data_t
#define data_t double
#endif /*data_t*/


class cudaMatrix
{
private:
    int _row,_col,_ld;
    
    //checker to avoid re-copy to dev. 
    bool locally_changed;


    // Two pointers: local and device
    data_t *host_data, *dev_data;
    void allocate(int row,int col);
    void deallocate(int row,int col);
    int MemSize(bool bin);

public:
    cudaMatrix(int row, int col);
    int size(bool bin);

    cudaMatrix();
    ~cudaMatrix();
};



class GPU_properties{
    // for future: include multiples GPUS
};

class Event_Handler
{
    // TODO: How to synch ??????
};




#endif /*_CUDAMATRIX_HPP__*/