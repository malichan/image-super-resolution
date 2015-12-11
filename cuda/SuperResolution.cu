#include "SuperResolution.cuh"

template <>
HostMatrix SuperResolution<HostMatrix>::reconstructPatches(const HostMatrix& patchesHigh,
    unsigned int height, unsigned int width) {
    unsigned int patchesVertical = (height - 3 + 5) / 6;
    unsigned int patchesHorizontal = (width - 3 + 5) / 6;

    HostMatrix sumMatrix(height, width);
    HostMatrix countMatrix(height, width);
    MatrixOperations<HostMatrix>::fill(sumMatrix, 0.0f);
    MatrixOperations<HostMatrix>::fill(countMatrix, 0.0f);

    for (unsigned int pi = 0; pi < patchesVertical; ++pi) {
        for (unsigned int pj = 0; pj < patchesHorizontal; ++pj) {
            unsigned int pIndexJ = pi * patchesHorizontal + pj;
            for (unsigned int i = 0; i < 9; ++i) {
                unsigned int iIndexI = pi * 6 + i;
                for (unsigned int j = 0; j < 9; ++j) {
                    unsigned int pIndexI = i * 9 + j;
                    unsigned int iIndexJ = pj * 6 + j;
                    if (iIndexI < height && iIndexJ < width) {
                        sumMatrix.setElement(iIndexI, iIndexJ,
                            sumMatrix.getElement(iIndexI, iIndexJ) +
                            patchesHigh.getElement(pIndexI, pIndexJ));
                        countMatrix.setElement(iIndexI, iIndexJ,
                            countMatrix.getElement(iIndexI, iIndexJ) + 1.0f);
                    }
                }
            }
        }
    }

    HostMatrix imageHigh = MatrixOperations<HostMatrix>::divide(sumMatrix, countMatrix);
    return imageHigh;
}

__global__
void _reconstructPatches(const float* patches, float* sum, float* count,
    unsigned int m_patches, unsigned int n_patches, unsigned int m_image, unsigned int n_image,
    unsigned int patches_h) {
    unsigned int global_i = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int global_j = threadIdx.x + blockIdx.x * blockDim.x;

    if (global_i < m_patches && global_j < n_patches) {
        unsigned int image_i = (global_j / patches_h) * 6 + (global_i / 9);
        unsigned int image_j = (global_j % patches_h) * 6 + (global_i % 9);

        if (image_i < m_image && image_j < n_image) {
            atomicAdd(&sum[image_i * n_image + image_j], patches[global_i * n_patches + global_j]);
            atomicAdd(&count[image_i * n_image + image_j], 1.0f);
        }
    }
}

template <>
DeviceMatrix SuperResolution<DeviceMatrix>::reconstructPatches(const DeviceMatrix& patchesHigh,
    unsigned int height, unsigned int width) {
    const unsigned int BLOCK_SIZE = 32;

    unsigned int patchesHorizontal = (width - 3 + 5) / 6;

    DeviceMatrix sumMatrix(height, width);
    DeviceMatrix countMatrix(height, width);
    MatrixOperations<DeviceMatrix>::fill(sumMatrix, 0.0f);
    MatrixOperations<DeviceMatrix>::fill(countMatrix, 0.0f);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((patchesHigh.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (patchesHigh.getHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    _reconstructPatches<<<gridDim, blockDim>>>(patchesHigh.getElements(),
        sumMatrix.getElements(), countMatrix.getElements(), patchesHigh.getHeight(), patchesHigh.getWidth(),
        height, width, patchesHorizontal);

    DeviceMatrix imageHigh = MatrixOperations<DeviceMatrix>::divide(sumMatrix, countMatrix);
    return imageHigh;
}