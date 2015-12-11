#ifndef _SUPERRESOLUTION_CUH_
#define _SUPERRESOLUTION_CUH_

#include "CoupledDict.cuh"
#include "NeuralNet.cuh"

template <typename MatrixType>
class SuperResolution {
public:
    SuperResolution();

    MatrixType scaleUp3x(const MatrixType& imageLow) const;

private:
    CoupledDict<MatrixType> coupledDict;
    NeuralNet<MatrixType> neuralNet;

    static MatrixType decomposePatches(const MatrixType& imageLow) {return imageLow;}
    static MatrixType reconstructPatches(const MatrixType& patchesHigh,
        unsigned int height, unsigned int width);
    static void globalOptimize(MatrixType& imageHigh, const MatrixType& imageLow) {}
};

#include "SuperResolution.cut"

#endif