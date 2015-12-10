#ifndef _BINARYOPERATIONS_CUH_
#define _BINARYOPERATIONS_CUH_

#include "PreCompile.cuh"

class Add {
public:
    __host__ __device__
    float operator()(float inA, float inB) const {
        return inA + inB;
    }
};

class Multiply {
public:
    __host__ __device__
    float operator()(float inA, float inB) const {
        return inA * inB;
    }
};

#endif