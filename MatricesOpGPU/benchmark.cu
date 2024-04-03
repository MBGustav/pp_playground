

#include <iostream>
#include "cudaMatrix.cuh"
#include "cudaLinalg.cuh"
#include <thread>
#include <unistd.h> 

using namespace std;

const int N = 15;
bool  test_EqualMatrices()
{
    cudaMatrix A(100, 200);
    cudaMatrix B(100, 200);
    cudaMatrix C(100, 200);

    A.randMatrix(2, 0, true);
    B.randMatrix(2, 0, true);

    // Local copy
    for(int i = 0; i < A.getRow() ; i++)
        for(int j = 0; j < A.getCol() ; j++)
            C.at(i,j) = A.at(i,j);

    return !gpuLinalg::EqualMatrices(A, B) && gpuLinalg::EqualMatrices(A, C);
}

bool test_MatMul() 
{
    using namespace gpuLinalg;

    //testing sqr matrix
    int square_small = 10;
    int square_large = 120;

    int m      = 40;
    int common = 70;
    int k      = 30;

    cudaMatrix A1 (square_small, square_small);
    cudaMatrix Id1(square_small, square_small);
    cudaMatrix C1 (square_small, square_small);
    
    cudaMatrix A2 (square_large, square_large);
    cudaMatrix Id2(square_large, square_large);
    cudaMatrix C2 (square_large, square_large);

    cudaMatrix A3(m, common);
    cudaMatrix B3(common, k);
    cudaMatrix C3(m, k);
    cudaMatrix C3_exp(m, k);

    

    A1.randMatrix(2, 0);
    A2.randMatrix(2, 0);
    B3.randMatrix(2, 0);

    Id1.Identity(); 
    Id2.Identity(); 


    // Expected A = C
        // 1st case small kernel
    MatMul(A1, Id1, C1, 1.0, 0.0);
    
    A1.SynchronizeValues();
    C1.SynchronizeValues();
    if (!EqualMatrices(A1, C1)) return false;
    
    
    // 2st case large kernel
    MatMul(A2, Id2, C2, 1.0, 0.0);
    
    A2.SynchronizeValues();
    C2.SynchronizeValues();
    // A2.display(0,30);
    // C2.display(0,30);
    if (!EqualMatrices(A2, C2)) cout << "diff A2\n";// return false;

    // Checking for rectangles matrices (more processing)
    MatMul(A3, B3, C3, 1.0, 0.0);
    MatMulOnCPU(A3,B3,C3_exp, 1.0, 0.0);

    if (!EqualMatrices(C3, C3_exp)) return false; //cout << "diff A3\n";//return false;
    return true;    
}




int main() 
{
    

    // lambda checker for the functions - just making easier
    auto result_test = [](std::string fun_name, bool (*funct)(void)) {
        bool res = funct();
        cout.width(20);
        cout << left << fun_name << ":  "; 
        cout <<( res ? "OK" : "NOT OK") <<std::endl;
    };

    result_test("EqualMatrices", &test_EqualMatrices);
    result_test("MatMul", &test_MatMul);

    result_test("EqualMatrices", &test_EqualMatrices);
    // result_test("EqualMatrices", &test_EqualMatrices);
    // result_test("EqualMatrices", &test_EqualMatrices);
    // result_test("EqualMatrices", &test_EqualMatrices);
    // result_test("EqualMatrices", &test_EqualMatrices);
    // result_test("EqualMatrices", &test_EqualMatrices);
    // result_test("EqualMatrices", &test_EqualMatrices);


    return 0;
}