#ifndef _MATRIX_CUH_
#define _MATRIX_CUH_

#include "PreCompile.cuh"

class Matrix {
public:
    unsigned int getHeight() const {
        return height;
    }
    unsigned int getWidth() const {
        return width;
    }
    bool isOnDevice() const {
        return onDevice;
    }
    float* getElements() const {
        return elements.get();
    }

    virtual float getElement(unsigned int i, unsigned int j) const = 0;
    virtual void setElement(unsigned int i, unsigned int j, float value) = 0;

protected:
    unsigned int height;
    unsigned int width;
    bool onDevice;
    std::shared_ptr<float> elements;

    Matrix(unsigned int height, unsigned int width, bool onDevice):
        height(height), width(width), onDevice(onDevice), elements(0) {}
};

class HostMatrix : public Matrix {
public:
    HostMatrix(): Matrix(0, 0, false) {}
    HostMatrix(unsigned int height, unsigned int width);

    virtual float getElement(unsigned int i, unsigned int j) const;
    virtual void setElement(unsigned int i, unsigned int j, float value);
};

class DeviceMatrix : public Matrix {
public:
    DeviceMatrix(): Matrix(0, 0, true) {}
    DeviceMatrix(unsigned int height, unsigned int width);

    virtual float getElement(unsigned int i, unsigned int j) const;
    virtual void setElement(unsigned int i, unsigned int j, float value);
};

#endif