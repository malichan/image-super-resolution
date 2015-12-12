#ifndef _BITMAP_CUH_
#define _BITMAP_CUH_

#include "Matrix.cuh"

class Bitmap {
public:
    static HostMatrix loadFromFile(const char* fileName);
    static void saveToFile(const HostMatrix& image, const char* fileName);

private:
    static float convertToFloat(unsigned int value);
    static unsigned int convertToByte(float value);
};

inline float Bitmap::convertToFloat(unsigned int value) {
    return (float)value / 255.0f;
}

inline unsigned int Bitmap::convertToByte(float value) {
    if (value < 0.0f) value = 0.0f;
    if (value > 1.0f) value = 1.0f;
    return (unsigned int)roundf(value * 255.0f);
}

#endif