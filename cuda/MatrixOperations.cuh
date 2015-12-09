#ifndef _MATRIXOPERATIONS_CUH_
#define _MATRIXOPERATIONS_CUH_

#include "Matrix.cuh"

template <typename MatrixType>
class MatrixOperations {
public:
    template <typename UnaryOperation>
    static void transform(MatrixType& matrix, UnaryOperation op);

    static void fill(MatrixType& matrix, float value);

    static MatrixType multiply(const MatrixType& matrixA, const MatrixType& matrixB);

    // static MatrixType pad(const MatrixType& matrix, unsigned int height, unsigned int width,
    //     unsigned int i, unsigned int j, float value);

    static MatrixType transpose(const MatrixType& matrix);

    static MatrixType concatenateVertical(const MatrixType& matrixUpper, const MatrixType& matrixLower);
};

#include "MatrixOperations.cut"

#endif