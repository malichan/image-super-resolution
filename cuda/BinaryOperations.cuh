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

class Subtract {
public:
    __host__ __device__
    float operator()(float inA, float inB) const {
        return inA - inB;
    }
};

class Multiply {
public:
    __host__ __device__
    float operator()(float inA, float inB) const {
        return inA * inB;
    }
};

class Divide {
public:
    __host__ __device__
    float operator()(float inA, float inB) const {
        return inA / inB;
    }
};

class SquareDiff {
public:
    __host__ __device__
    float operator()(float inA, float inB) const {
        return (inA - inB) * (inA - inB);
    }
};

#endif