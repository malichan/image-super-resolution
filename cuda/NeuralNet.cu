#include "NeuralNet.cuh"

template <>
void NeuralNet<HostMatrix>::loadWeights(const char* weightsInFileName, const char* weightsOutFileName) {
    weightsIn = MatrixUtilities::loadFromFile(weightsInFileName);
    weightsOut = MatrixUtilities::loadFromFile(weightsOutFileName);
    initialized = true;
}

template <>
void NeuralNet<DeviceMatrix>::loadWeights(const char* weightsInFileName, const char* weightsOutFileName) {
    weightsIn = MatrixUtilities::copyToDevice(MatrixUtilities::loadFromFile(weightsInFileName));
    weightsOut = MatrixUtilities::copyToDevice(MatrixUtilities::loadFromFile(weightsOutFileName));
    initialized = true;
}