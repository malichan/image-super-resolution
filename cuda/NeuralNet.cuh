#ifndef _NEURALNET_CUH_
#define _NEURALNET_CUH_

#include "Matrix.cuh"

template <typename MatrixType>
class NeuralNet {
public:
    NeuralNet(): weightsIn(), weightsOut(), initialized(false) {}

    void loadWeights(const char* weightsInFileName, const char* weightsOutFileName);
    MatrixType predict(const MatrixType& instances);

private:
    MatrixType weightsIn;
    MatrixType weightsOut;
    bool initialized;
};

class Sigmoid {
public:
    __host__ __device__
    float operator()(float in) const {
        return 1.0f / (1.0f + expf(-in));
    }
};

#include "NeuralNet.cut"

#endif