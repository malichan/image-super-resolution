#ifndef _UNARYOPERATIONS_CUH_
#define _UNARYOPERATIONS_CUH_

#include "PreCompile.cuh"

class Identity {
public:
    __host__ __device__
    float operator()(float in) const {
        return in;
    }
};

class Negate {
public:
    __host__ __device__
    float operator()(float in) const {
        return -in;
    }
};

class Fill {
public:
    float value;

    Fill(float value): value(value) {}

    __host__ __device__
    float operator()(float in) const {
        return value;
    }
};

class Sqaure {
public:
    __host__ __device__
    float operator()(float in) const {
        return in * in;
    }
};

class Sigmoid {
public:
    __host__ __device__
    float operator()(float in) const {
        return 1.0f / (1.0f + expf(-in));
    }
};

#endif