#include <iostream>
#include "cudaMatrix.h"

// template<typename TT>
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
void cudaMatrix::display() const
{
    std::cout << " showing top left from Matrix:\n";
    for (int row = 0; row < std::min(MAX_DISPLAY_MATRIX, this->getRow()); row++) {
    for (int col = 0; col < std::min(MAX_DISPLAY_MATRIX, this->getCol()); col++) {
      // Copy old state of cout
      std::ios oldState(nullptr);
      oldState.copyfmt(std::cout);

      // Edit the output format of cout
      std::cout << std::fixed << std::setprecision(2);

      // Print the results
      std::cout << std::setw(8) << this->const_at(col, row) << " ";

      // Restore the output format of cout
      std::cout.copyfmt(oldState);
    }
    std::cout << std::endl;
  }
}

// template<typename TT>
__device__ __host__ data_t& cudaMatrix::data()
{
#ifdef __CUDA_ARCH__
    return *dev_data;
#else
    return *host_data;
#endif
}

// template<typename TT>
__device__ __host__ data_t& cudaMatrix::at(int x, int y)
{
#ifdef __CUDA_ARCH__
    return dev_data[idx_matrix(_ld, x, y)];
#else
    return host_data[idx_matrix(_ld, x, y)];
    Change=ChangeOnHost;// is this the best way ? 
#endif
}

// template<typename TT>
__device__ __host__ data_t& cudaMatrix::const_at(int x, int y) const
{
#ifdef __CUDA_ARCH__
    return dev_data[idx_matrix(_ld, x, y)];
#else
    return host_data[idx_matrix(_ld, x, y)];
#endif
}


// template<typename TT>
bool cudaMatrix::TransferData(cudaMemcpyKind kind)
{   
    cudaMemcpy(dev_data, host_data, this->MemSize(true), kind);
    check_last_error();
    Change = Equal;
    return true;
}

// Returns row value
// template<typename TT>
int cudaMatrix::getRow() const { return this->_row;}

// Returns Col value 
// template<typename TT>
int cudaMatrix::getCol() const { return this->_col;}

// Returns Leading Dimension width
// template<typename TT>
int cudaMatrix::getLd() const { return this->_ld;}

// template<typename TT>
int cudaMatrix::getSize() const { return this->getCol() * this->getRow();}

// template<typename TT>
size_t cudaMatrix::MemSize(bool bin=false) const{
    return getLd() * getRow() * (bin ? sizeof(data_t) : 1 );
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
void cudaMatrix::randMatrix(data_t lower_bound, data_t upper_bound, bool Synchronize = true)
{
    int ld = getLd();
    for(int ii=0; ii< getRow(); ii++)
    for(int jj=0; jj< getCol(); jj++)
        this->at(ii,jj) = 
            static_cast<data_t>(rand()) / 
            static_cast<data_t>(RAND_MAX / (upper_bound - lower_bound)) + lower_bound ;

    Change = Synchronize ? Equal : ChangeOnHost;

    if(Synchronize) SynchronizeValues();
}

// template<typename TT>
void cudaMatrix::SynchronizeValues()
{
    bool sync = false;
    switch(Change){
        case ChangeOnHost: {
            sync = TransferData(cudaMemcpyHostToDevice);break;
        }
        case ChangeOnDevice : {
            sync = TransferData(cudaMemcpyDeviceToHost);break;
        }
        // default : data already sync. 
    }
    // if didnt sync, keep previous value from change
    this->Change = (sync) ? this->Change : Equal;
}

