#ifndef _BINARYINDEXEDOPERATIONS_CUH_
#define _BINARYINDEXEDOPERATIONS_CUH_

struct IndexedValue {
    float value;
    unsigned int index;

    __host__ __device__
    IndexedValue(float value, unsigned int index): value(value), index(index) {}
};

class MinIndexed {
public:
    __host__ __device__
    IndexedValue operator()(IndexedValue inA, IndexedValue inB) const {
        return (inA.value < inB.value)? inA: inB;
    }
};

#endif