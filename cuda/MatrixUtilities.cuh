#ifndef _MATRIXUTILITIES_CUH_
#define _MATRIXUTILITIES_CUH_

#include "Matrix.cuh"

class MatrixUtilities {
public:
    static HostMatrix loadFromFile(const char* fileName);
    static void saveToFile(const HostMatrix& matrix, const char* fileName);

    static bool compare(const HostMatrix& matrixA, const HostMatrix& matrixB, float epsilon);

    static HostMatrix copyToHost(const Matrix& matrix);
    static DeviceMatrix copyToDevice(const Matrix& matrix);

    template <typename MatrixType>
    static MatrixType copy(const MatrixType& matrix);
};

#endif