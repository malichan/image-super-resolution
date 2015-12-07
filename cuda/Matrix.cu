#include <cmath>
#include <fstream>

#include "Matrix.cuh"

HostMatrix::HostMatrix(unsigned int height, unsigned int width): Matrix(height, width) {
    cudaMallocHost(&elements, height * width * sizeof(float));
}

HostMatrix::~HostMatrix() {
    cudaFreeHost(elements);
}

float HostMatrix::getElement(unsigned int i, unsigned int j) const {
    if (i < height && j < width) {
        return elements[i * width + j];
    } else {
        return NAN;
    }
}

DeviceMatrix::DeviceMatrix(unsigned int height, unsigned int width): Matrix(height, width) {
    cudaMalloc(&elements, height * width * sizeof(float));
}

DeviceMatrix::~DeviceMatrix() {
    cudaFree(elements);
}

float DeviceMatrix::getElement(unsigned int i, unsigned int j) const {
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
    for (unsigned int i = 0; i < height * width; ++i) {
        fin >> matrix.elements[i];
    }

    fin.close();
    return matrix;
}

HostMatrix MatrixUtilities::copyToHost(const DeviceMatrix& matrix) {
    HostMatrix copy_matrix(matrix.height, matrix.width);
    cudaMemcpy(copy_matrix.elements, matrix.elements, matrix.height * matrix.width * sizeof(float), cudaMemcpyDeviceToHost);
    return copy_matrix;
}

DeviceMatrix MatrixUtilities::copyToDevice(const HostMatrix& matrix) {
    DeviceMatrix copy_matrix(matrix.height, matrix.width);
    cudaMemcpy(copy_matrix.elements, matrix.elements, matrix.height * matrix.width * sizeof(float), cudaMemcpyHostToDevice);
    return copy_matrix;
}