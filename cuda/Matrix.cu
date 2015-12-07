#include <cmath>
#include <fstream>
#include <iomanip>

#include "Matrix.cuh"

inline HostMatrix::HostMatrix(unsigned int height, unsigned int width): Matrix(height, width, false) {
    cudaMallocHost(&elements, height * width * sizeof(float));
}

inline HostMatrix::~HostMatrix() {
    cudaFreeHost(elements);
}

inline float HostMatrix::getElement(unsigned int i, unsigned int j) const {
    if (i < height && j < width) {
        return elements[i * width + j];
    } else {
        return NAN;
    }
}

inline DeviceMatrix::DeviceMatrix(unsigned int height, unsigned int width): Matrix(height, width, true) {
    cudaMalloc(&elements, height * width * sizeof(float));
}

inline DeviceMatrix::~DeviceMatrix() {
    cudaFree(elements);
}

inline float DeviceMatrix::getElement(unsigned int i, unsigned int j) const {
    if (i < height && j < width) {
        float elem = 0.0f;
        cudaMemcpy(&elem, &elements[i * width + j], sizeof(float), cudaMemcpyDeviceToHost);
        return elem;
    } else {
        return NAN;
    }
}

HostMatrix MatrixUtilities::loadFromFile(const char* fileName) {
    std::ifstream fin;
    fin.open(fileName);

    unsigned int height = 0;
    unsigned int width = 0;
    fin >> height >> width;

    HostMatrix matrix(height, width);
    for (unsigned int k = 0; k < height * width; ++k) {
        fin >> matrix.elements[k];
    }

    fin.close();
    return matrix;
}

void MatrixUtilities::saveToFile(const HostMatrix& matrix, const char* fileName) {
    std::ofstream fout;
    fout.open(fileName);

    fout.setf(std::ios::fixed, std::ios::floatfield);
    fout.precision(6);
    fout << matrix.height << " " << matrix.width << std::endl;
    for (unsigned int i = 0; i < matrix.height; ++i) {
        for (unsigned int j = 0; j < matrix.width; ++j) {
            fout << matrix.getElement(i, j) << " ";
        }
        fout << std::endl;
    }

    fout.close();
}

HostMatrix MatrixUtilities::copyToHost(const Matrix& matrix) {
    HostMatrix copy_matrix(matrix.height, matrix.width);
    size_t count = matrix.height * matrix.width * sizeof(float);
    cudaMemcpyKind kind = matrix.onDevice? cudaMemcpyDeviceToHost: cudaMemcpyHostToHost;
    cudaMemcpy(copy_matrix.elements, matrix.elements, count, kind);
    return copy_matrix;
}

DeviceMatrix MatrixUtilities::copyToDevice(const Matrix& matrix) {
    DeviceMatrix copy_matrix(matrix.height, matrix.width);
    size_t count = matrix.height * matrix.width * sizeof(float);
    cudaMemcpyKind kind = matrix.onDevice? cudaMemcpyDeviceToDevice: cudaMemcpyHostToDevice;
    cudaMemcpy(copy_matrix.elements, matrix.elements, count, kind);
    return copy_matrix;
}