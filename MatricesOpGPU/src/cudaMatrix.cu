#include <iostream>
#include "cudaMatrix.cuh"

// template<typename TT>
void cudaMatrix::allocate(int row,int col)
{
    this->_row = row;
    this->_col = col;
    // to make padding efficient
    this->_ld  = ld_padding(col);

    // TODO: should I implement here te Async Memory, or alloc through the op? No..
    
    MALLOC(host_data, this->MemSize(true));

    cudaMalloc((void **)&dev_data, this->MemSize(true));

    check_last_error();
    is_transposed=false;
    Change = Equal;
}

// template<typename TT>
void cudaMatrix::deallocate()
{
    free(host_data);
    cudaFree(dev_data);
    check_last_error ();
}

// template<typename TT>
cudaMatrix::cudaMatrix() : host_data{NULL}, dev_data{NULL}{}

// template<typename TT>
cudaMatrix::cudaMatrix(int row, int col) : _row(row), _col(col){allocate(row, col);}

// template<typename TT>
cudaMatrix::~cudaMatrix() {deallocate();}

// template<typename TT>
void cudaMatrix::display(int off_x, int off_y)
{
    SynchronizeValues();
    std::cout << " showing top left from Matrix:\n";
    for (int row = off_x; row < std::min(off_x + MAX_DISPLAY_MATRIX, this->getRow()); row++) {
    for (int col = off_y; col < std::min(off_y + MAX_DISPLAY_MATRIX, this->getCol()); col++) {
      // Copy old state of cout
      std::ios oldState(nullptr);
      oldState.copyfmt(std::cout);

      // Edit the output format of cout
      std::cout << std::fixed << std::setprecision(2);

      // Print the results
      std::cout << std::setw(8) << this->at(row, col) << " ";

      // Restore the output format of cout
      std::cout.copyfmt(oldState);
    }
    std::cout << std::endl;
  }
}

// template<typename TT>
data_t* cudaMatrix::data()
{
    return host_data;
}

// data_t* cudaMatrix::data()
// {
//     return host_data;
// }


// data_t* cudaMatrix::dataGPU()
// {
//     return dev_data;
// }

data_t* cudaMatrix::dataGPU()
{
    return (this->dev_data);
}


// template<typename TT>
data_t& cudaMatrix::at(int x, int y)
{
#ifdef __CUDA_ARCH__
    return dev_data[idx_matrix(_ld, x, y)];
#else
    return host_data[idx_matrix(_ld, x, y)];
#endif
}

// template<typename TT>
data_t& cudaMatrix::const_at(int x, int y) const
{
#ifdef __CUDA_ARCH__
    return dev_data[idx_matrix(_ld, x, y)];
#else
    return host_data[idx_matrix(_ld, x, y)];
#endif
}


// Returns row value
// template<typename TT>
int cudaMatrix::getRow() const { return this->_row;}

// Returns Col value 
// template<typename TT>
int cudaMatrix::getCol() const { return this->_col;}

// Returns Leading Dimension width
// template<typename TT>
int cudaMatrix::get_ld() const { return this->_ld;}

// template<typename TT>
int cudaMatrix::getSize() const { return this->getCol() * this->getRow();}

// template<typename TT>
size_t cudaMatrix::MemSize(bool bin=false) const{
    return get_ld() * getRow() * (bin ? sizeof(data_t) : 1 );
}

//This case really forces to transpose data
// template<typename TT>
void cudaMatrix::transposeInPlace()
{
    
}

// returns a matrix with the same configuration transposed
// template<typename TT>
cudaMatrix cudaMatrix::transpose()
{
    cudaMatrix result(this->getCol(), this->getRow());

    
    #pragma omp parallel for
    for(int i=0; i< this->getCol();i++)
        for(int j=0; j< this->getCol();j++)
            result.at(i,j) = this->at(j,i);
    
    return result;

}

//param: out(bool) check if offload executed correctly
// template<typename TT>
/*
    Generate a Random Matrix 
        input: lower_bound, upper_bound [data_t]
        input: Synchronize [bool]
    Synch is used to sync between GPU and CPU
*/      
void cudaMatrix::randMatrix(data_t lower_bound, data_t upper_bound, bool Synchronize)
{
    Change = ChangeOnHost;
    int ld = get_ld();
    for(int ii=0; ii< getRow(); ii++)
    for(int jj=0; jj< getCol(); jj++)
        this->at(ii,jj) = 
            static_cast<data_t>(rand()) / 
            static_cast<data_t>(RAND_MAX / (upper_bound - lower_bound)) + lower_bound ;

    if(Synchronize) SynchronizeValues();
}

void cudaMatrix::Identity(data_t val, bool Synchronize)
{
    Change = ChangeOnHost;
    for(int ii=0; ii< getRow(); ii++)
    for(int jj=0; jj< getCol(); jj++)
        this->at(ii,jj) = ii == jj ? val : 0.0f;
    
    if(Synchronize) SynchronizeValues();
}


void cudaMatrix::changeOccurred(ChangeHandler Status){this->Change = Status;}


// template<typename TT>
bool cudaMatrix::TransferData(cudaMemcpyKind kind)
{   
    if(kind == cudaMemcpyHostToDevice)
        cudaMemcpy(dev_data, host_data, this->MemSize(true), kind);
    else
        cudaMemcpy(host_data, dev_data, this->MemSize(true), kind);
    
    check_last_error();
    Change = Equal;
    return true;
}

// template<typename TT>
void cudaMatrix::SynchronizeValues()
{
    bool sync = false;
    switch(Change){
        case ChangeOnHost: {
            sync = TransferData(cudaMemcpyHostToDevice);
            break;
        }
        case ChangeOnDevice : {
            sync = TransferData(cudaMemcpyDeviceToHost);
            break;              
        }
        // default : data already sync. 
    }
    // if didnt sync, keep previous value from change
    this->Change = (sync) ? this->Change : Equal;
}

