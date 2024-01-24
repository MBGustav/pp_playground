

#include <iostream>
#include "cudaMatrix.h"
#include "cudaLinalg.h"

int main() 
{
    cudaMatrix A(3, 3), B(3, 3), C(3,3);
    A.randMatrix(6, 6,true);
    A.display();

    B.randMatrix(6, 6,true);
    B.display(); 


    gpuLinalg::MatMul(A,B,C, 1.0, 0.0);

    C.display();


    


    return 0;
}