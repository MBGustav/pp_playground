

#include <iostream>
#include "cudaMatrix.cuh"
#include "cudaLinalg.cuh"

int main() 
{
    cudaMatrix A(10, 10), B(10, 10), C(10,10);
    A.randMatrix(6, 6, true);
    A.display();

    B.randMatrix(6, 6, true);
    B.display(); 

    std:: cout << "sizeof" << sizeof(cudaMatrix) << std::endl;
    gpuLinalg::MatMul(A,B,C, 2.0, 0.0);

    C.display();


    


    return 0;
}