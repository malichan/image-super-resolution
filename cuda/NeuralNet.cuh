#ifndef _NEURALNET_CUH_
#define _NEURALNET_CUH_

#include "MatrixUtilities.cuh"
#include "MatrixOperations.cuh"

template <typename MatrixType>
class NeuralNet {
public:
    NeuralNet(): weightsIn(), weightsOut(), initialized(false) {}

    void loadWeights(const char* weightsInFileName, const char* weightsOutFileName);
    MatrixType predict(const MatrixType& instances) const;

private:
    MatrixType weightsIn;
    MatrixType weightsOut;
    bool initialized;
};

#include "NeuralNet.cut"

#endif