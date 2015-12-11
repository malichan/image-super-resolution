#ifndef _MATRIXOPERATIONS_CUH_
#define _MATRIXOPERATIONS_CUH_

#include "Matrix.cuh"
#include "UnaryOperations.cuh"
#include "BinaryOperations.cuh"
#include "BinaryIndexedOperations.cuh"

template <typename MatrixType>
class MatrixOperations {
public:
    template <typename UnaryOperation>
    static void transform(MatrixType& matrix, UnaryOperation op);
    static void fill(MatrixType& matrix, float value);
    static void negate(MatrixType& matrix);

    template <typename BinaryOperation>
    static MatrixType combineElementWise(const MatrixType& matrixA, const MatrixType& matrixB,
        BinaryOperation op);
    static MatrixType add(const MatrixType& matrixA, const MatrixType& matrixB);
    static MatrixType subtract(const MatrixType& matrixA, const MatrixType& matrixB);
    static MatrixType divide(const MatrixType& matrixA, const MatrixType& matrixB);

    template <typename BinaryOperation1, typename BinaryOperation2, typename UnaryOperation>
    static MatrixType combineInnerProduct(const MatrixType& matrixA, const MatrixType& matrixB,
        BinaryOperation1 opA, BinaryOperation2 opB, UnaryOperation opC, float identity);
    static MatrixType multiply(const MatrixType& matrixA, const MatrixType& matrixB);

    static MatrixType transpose(const MatrixType& matrix);

    static MatrixType padRowsTop(const MatrixType& matrix, unsigned int rows, float value);
    static MatrixType concatenateRows(const MatrixType& matrixUpper, const MatrixType& matrixLower);

    template <typename BinaryOperation, typename UnaryOperation>
    static MatrixType reduceColumns(const MatrixType& matrix,
        BinaryOperation opA, UnaryOperation opB, float identity);
    static MatrixType sumColumns(const MatrixType& matrix);

    template <typename BinaryIndexedOperation>
    static MatrixType reduceColumnsIndexed(const MatrixType& matrix,
        BinaryIndexedOperation op, IndexedValue identity);
    static MatrixType minColumnsIndexed(const MatrixType& matrix);

    template <typename BinaryOperation1, typename UnaryOperation, typename BinaryOperation2>
    static void reduceTransformColumns(MatrixType& matrix,
        BinaryOperation1 opA, UnaryOperation opB, BinaryOperation2 opC, float identity);
    static void normalizeColumns(MatrixType& matrix);

    static MatrixType index(const MatrixType& matrix, const MatrixType& vector);
};

#include "MatrixOperations.cut"

#endif