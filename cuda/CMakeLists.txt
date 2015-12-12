cmake_minimum_required(VERSION 2.8)
project(Super_Resolution_3x)

include(ParallelUtils.cmake)
enable_cuda_support()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")

cuda_add_executable(sr3x Matrix.cu MatrixUtilities.cu MatrixOperations.cu Bitmap.cu CoupledDict.cu NeuralNet.cu SuperResolution.cu Timer.cu Main.cu)