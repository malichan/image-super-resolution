#include "Matrix.cuh"

class HostMatrixDeleter {
public:
    void operator()(float* ptr) const {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
};

HostMatrix::HostMatrix(unsigned int height, unsigned int width): Matrix(height, width, false) {
    float* rawElements = 0;
    cudaMallocHost(&rawElements, height * width * sizeof(float));
    elements = std::shared_ptr<float>(rawElements, HostMatrixDeleter());
}


float HostMatrix::getElement(unsigned int i, unsigned int j) const {
    if (i >= height || j >= width) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        return (elements.get())[i * width + j];
    }
}

void HostMatrix::setElement(unsigned int i, unsigned int j, float value) {
    if (i >= height || j >= width) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        (elements.get())[i * width + j] = value;
    }
}

class DeviceMatrixDeleter {
public:
    void operator()(float* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

DeviceMatrix::DeviceMatrix(unsigned int height, unsigned int width): Matrix(height, width, true) {
    float* rawElements = 0;
    cudaMalloc(&rawElements, height * width * sizeof(float));
    elements = std::shared_ptr<float>(rawElements, DeviceMatrixDeleter());
}

float DeviceMatrix::getElement(unsigned int i, unsigned int j) const {
    if (i >= height || j >= width) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        float value = 0.0f;
        cudaMemcpy(&value, elements.get() + i * width + j, sizeof(float), cudaMemcpyDeviceToHost);
        return value;
    }
}

void DeviceMatrix::setElement(unsigned int i, unsigned int j, float value) {
    if (i >= height || j >= width) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        cudaMemcpy(elements.get() + i * width + j, &value, sizeof(float), cudaMemcpyHostToDevice);
    }
}