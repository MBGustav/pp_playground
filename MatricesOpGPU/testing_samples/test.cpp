#include <iostream>
// #include "cudaMatrix.h"


#define ROW 3
#define COL 3

#define idx_matrix(i, j, ld, transposed) ((transposed) ? ((j) * (ld) + (i)) : ((i) * (ld) + (j)))

// Function to print a matrix
void printMatrix(int *mat, int row, int col, int transposed) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", mat[idx_matrix(i, j, col, transposed)]);
        }
        printf("\n");
    }
}

int main() {
    int matrix[ROW][COL] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int transpose = 1; // Change this to 0 for original matrix, 1 for transposed matrix

    if (transpose) {
        printf("Transposed Matrix:\n");
        printMatrix((int *)matrix, ROW, COL, 1);
    } else {
        printf("Original Matrix:\n");
        printMatrix((int *)matrix, ROW, COL, 0);
    }

    return 0;
}


// int main() 
// {

// 	cudaMatrix A(9, 9);
// 	cudaMatrix B(9, 9);
// 	cudaMatrix C(9, 9);

	
// 	A.randMatrixOnHost();
// 	B.randMatrixOnHost();
// 	C.randMatrixOnHost();

// 	Multiply(1.0, A,B, 2.0, C);
	
// 	C.display();

// 	return 0;
// }