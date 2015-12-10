#include "CoupledDict.cuh"

template <>
void CoupledDict<HostMatrix>::loadDicts(const char* dictHighFileName, const char* dictLowFileName) {
    dictHigh = MatrixUtilities::loadFromFile(dictHighFileName);
    dictLow = MatrixUtilities::loadFromFile(dictLowFileName);
    dictLow = MatrixOperations<HostMatrix>::transpose(dictLow);
    initialized = true;
}

template <>
void CoupledDict<DeviceMatrix>::loadDicts(const char* dictHighFileName, const char* dictLowFileName) {
    dictHigh = MatrixUtilities::copyToDevice(MatrixUtilities::loadFromFile(dictHighFileName));
    dictLow = MatrixUtilities::copyToDevice(MatrixUtilities::loadFromFile(dictLowFileName));
    dictLow = MatrixOperations<DeviceMatrix>::transpose(dictLow);
    initialized = true;
}