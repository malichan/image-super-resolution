#ifndef _MATRIX_CUH_
#define _MATRIX_CUH_

class MatrixUtilities;

class Matrix {
    friend class MatrixUtilities;

public:
    virtual ~Matrix() {}

    unsigned int getHeight() const {
        return height;
    }
    unsigned int getWidth() const {
        return width;
    }
    float* getElements() const {
        return elements;
    }

    virtual float getElement(unsigned int i, unsigned int j) const = 0;

protected:
    unsigned int height;
    unsigned int width;
    float* elements;
    bool onDevice;

    Matrix(unsigned int height, unsigned int width, bool onDevice):
        height(height), width(width), elements(0), onDevice(onDevice) {}
};

class HostMatrix : public Matrix {
public:
    HostMatrix(unsigned int height, unsigned int width);
    virtual ~HostMatrix();

    virtual float getElement(unsigned int i, unsigned int j) const;
};

class DeviceMatrix : public Matrix {
public:
    DeviceMatrix(unsigned int height, unsigned int width);
    virtual ~DeviceMatrix();

    virtual float getElement(unsigned int i, unsigned int j) const;
};

class MatrixUtilities {
public:
    static HostMatrix loadFromFile(const char* fileName);
    static void saveToFile(const HostMatrix& matrix, const char* fileName);

    static HostMatrix copyToHost(const Matrix& matrix);
    static DeviceMatrix copyToDevice(const Matrix& matrix);

    template <typename UnaryOperation>
    static void transformOnHost(HostMatrix& matrix, UnaryOperation op);
    template <typename UnaryOperation>
    static void transformOnDevice(DeviceMatrix& matrix, UnaryOperation op);
};

#include "Matrix.cut"

#endif