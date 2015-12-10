#ifndef _MATRIXOPERATIONS_CUH_
#define _MATRIXOPERATIONS_CUH_

#include "Matrix.cuh"
#include "UnaryOperations.cuh"
#include "BinaryOperations.cuh"

template <typename MatrixType>
class MatrixOperations {
public:
    static MatrixType deepCopy(const MatrixType& matrix);

    template <typename UnaryOperation>
    static void transform(MatrixType& matrix, UnaryOperation op);
    static void fill(MatrixType& matrix, float value);
    static void negate(MatrixType& matrix);

    template <typename BinaryOperation>
    static MatrixType combineElementWise(const MatrixType& matrixA, const MatrixType& matrixB,
        BinaryOperation op);
    static MatrixType add(const MatrixType& matrixA, const MatrixType& matrixB);

    template <typename BinaryOperation1, typename BinaryOperation2, typename UnaryOperation>
    static MatrixType combineInnerProduct(const MatrixType& matrixA, const MatrixType& matrixB,
        BinaryOperation1 opA, BinaryOperation2 opB, UnaryOperation opC, float identity);
    static MatrixType multiply(const MatrixType& matrixA, const MatrixType& matrixB);

    // static MatrixType pad(const MatrixType& matrix, unsigned int height, unsigned int width,
    //     unsigned int i, unsigned int j, float value);

    static MatrixType transpose(const MatrixType& matrix);

    static MatrixType concatenateRows(const MatrixType& matrixUpper, const MatrixType& matrixLower);

    template <typename BinaryOperation, typename UnaryOperation>
    static MatrixType reduceColumns(const MatrixType& matrix,
        BinaryOperation opA, UnaryOperation opB, float identity);
    static MatrixType sumColumns(const MatrixType& matrix);
};

#include "MatrixOperations.cut"

#endif