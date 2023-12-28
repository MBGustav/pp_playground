#include <iostream>
#include <iomanip>

#include "kernel.hpp"
#include "Matrix.hpp"

#define EPSILON (1e-2)
using namespace std;



void print_banner()
{
    std::cout << "Matricial Operations - Multiplication\n";
    std::cout << "Oper: A x B = C\n";

    std::cout << "Usage: ./main <m> <n> <k>\n";
    std::cout << "\t m - Rows of A\n";
    std::cout << "\t n - Cols of A and Rows of B\n";
    std::cout << "\t k - Cols of B\n";
    exit(EXIT_FAILURE);
}

void FillMatrix(Matrix &A, int max_val)
{
    for(int i=0; i < A.size(); i++)
        A.data[i] = ((data_t)rand() / (data_t) RAND_MAX) * max_val;
}


void MatMulRef(Matrix &A, Matrix &B, Matrix &C_ref)
{
    int kArows = A.Row;
    int kBcols = B.Col;
    int kCommon = B.Row;
    for(int i=0; i < kArows; i++){
            for(int j=0; j < kBcols; j++){
                data_t acc = 0.0f;
                for(int k=0; k<kCommon;k++)
                    acc += A.data[kCommon*i + k] * B.data[kBcols*k + j];
                C_ref.data[kBcols*i +j]  = acc;
            }

        }
}

bool CheckResults(Matrix &C,Matrix &C_ref)
{
    const data_t epsilon = EPSILON;
    int size = C_ref.size();
    for(int idx=0; idx<size;idx++)
        if(abs(C_ref.data[idx] - C.data[idx]) > epsilon) return false;
    return true;
}


int main(int argc, char **argv)
{   
    
    if(argc != 4)
        print_banner();

    //Collect matrices parameters
    int Arows =  atoi(argv[1]);
    int kCommon = atoi(argv[2]);
    int Bcols = atoi(argv[3]);


    //Device selection
    //We will explicitly compile for the FPGA_EMULATOR, CPU_HOST, or FPGA
    #if defined(FPGA_SIMULATOR)
     auto selector = sycl::ext::intel::fpga_simulator_selector_v;
    #elif defined(FPGA_HARDWARE)
     auto selector = sycl::fpga_selector_v;
    #else //FPGA_EMULATOR
     auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    #endif


    auto property_list = sycl::property_list{
                sycl::property::queue::enable_profiling()};


    sycl::queue queue = sycl::queue(selector, property_list);

    cout << "Running on: " 
         << queue.get_device().get_info<sycl::info::device::name>().c_str()
         << std::endl;
         

    //Set MAtrices Size
    Matrix MatrixA(Arows, kCommon);
    Matrix MatrixB(kCommon, Bcols);
    Matrix MatrixC(Arows, Bcols);
    Matrix MatrixC_ref(Arows, Bcols);
    
    // Set  Matrices PArams 
    data_t kRandMax = 127.0f;

    FillMatrix(MatrixA, kRandMax);
    FillMatrix(MatrixB, kRandMax);

    //Set Kernel Call - MatrixMultiplication
    MatrixMultiplication(queue, MatrixA, MatrixB, MatrixC);

    MatMulRef(MatrixA, MatrixB, MatrixC_ref);
    // Check Results
    std::cout << (CheckResults(MatrixC, MatrixC_ref) ? "Results PASSED" : "Results not PASSED" ) 
              << std::endl;
    



    return 0;
}