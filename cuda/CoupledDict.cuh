#ifndef _COUPLEDDICT_CUH_
#define _COUPLEDDICT_CUH_

#include "MatrixUtilities.cuh"
#include "MatrixOperations.cuh"

template <typename MatrixType>
class CoupledDict {
public:
    CoupledDict(): dictHigh(), dictLow(), initialized(false) {}

    void loadDicts(const char* dictHighFileName, const char* dictLowFileName);
    MatrixType lookup(const MatrixType& instances);

private:
    MatrixType dictHigh;
    MatrixType dictLow;
    bool initialized;
};

#include "CoupledDict.cut"

#endif