#include "SuperResolution.cuh"

template <>
HostMatrix SuperResolution<HostMatrix>::decomposePatches(const HostMatrix& imageLow) {
    unsigned int height = imageLow.getHeight();
    unsigned int width = imageLow.getWidth();
    unsigned int patchesVertical = (height - 1 + 1) / 2;
    unsigned int patchesHorizontal = (width - 1 + 1) / 2;

    HostMatrix patchesLow(9, patchesVertical * patchesHorizontal);

    for (unsigned int pi = 0; pi < patchesVertical; ++pi) {
        for (unsigned int pj = 0; pj < patchesHorizontal; ++pj) {
            unsigned int pIndexJ = pi * patchesHorizontal + pj;
            for (unsigned int i = 0; i < 3; ++i) {
                unsigned int iIndexI = pi * 2 + i;
                if (iIndexI >= height) {
                    iIndexI = height * 2 - iIndexI - 1;
                }
                for (unsigned int j = 0; j < 3; ++j) {
                    unsigned int pIndexI = i * 3 + j;
                    unsigned int iIndexJ = pj * 2 + j;
                    if (iIndexJ >= width) {
                        iIndexJ = width * 2 - iIndexJ - 1;
                    }
                    patchesLow.setElement(pIndexI, pIndexJ, imageLow.getElement(iIndexI, iIndexJ));
                }
            }
        }
    }

    return patchesLow;
}

__global__
void _decomposePatches(const float* image, float* patches,
    unsigned int m_image, unsigned int n_image, unsigned int n_patches,
    unsigned int patches_v, unsigned int patches_h) {
    unsigned int global_i = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int global_j = threadIdx.x + blockIdx.x * blockDim.x;

    if (global_i < patches_v && global_j < patches_h) {
        unsigned int offset_i = global_i * 2;
        unsigned int offset_j = global_j * 2;
        unsigned int patches_j = global_i * patches_h + global_j;
        for (unsigned int i = 0; i < 3; ++i) {
            unsigned int image_i = offset_i + i;
            if (image_i >= m_image) {
                image_i = m_image * 2 - image_i - 1;
            }
            for (unsigned int j = 0; j < 3; ++j) {
                unsigned int image_j = offset_j + j;
                if (image_j >= n_image) {
                    image_j = n_image * 2 - image_j - 1;
                }
                unsigned int patches_i = i * 3 + j;
                patches[patches_i * n_patches + patches_j] = image[image_i * n_image + image_j];
            }
        }
        
    }
}

template <>
DeviceMatrix SuperResolution<DeviceMatrix>::decomposePatches(const DeviceMatrix& imageLow) {
    unsigned int patchesVertical = (imageLow.getHeight() - 1 + 1) / 2;
    unsigned int patchesHorizontal = (imageLow.getWidth() - 1 + 1) / 2;

    DeviceMatrix patchesLow(9, patchesVertical * patchesHorizontal);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((patchesHorizontal + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (patchesVertical + BLOCK_SIZE - 1) / BLOCK_SIZE);
    _decomposePatches<<<gridDim, blockDim>>>(imageLow.getElements(), patchesLow.getElements(),
        imageLow.getHeight(), imageLow.getWidth(), patchesLow.getWidth(),
        patchesVertical, patchesHorizontal);

    return patchesLow;
}

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

template <>
void SuperResolution<HostMatrix>::globalOptimize(HostMatrix& imageHigh, const HostMatrix& imageLow) {
    for (unsigned int i = 0; i < imageLow.getHeight(); ++i) {
        for (unsigned int j = 0; j < imageLow.getWidth(); ++j) {
            float valueLow = imageLow.getElement(i, j);
            float valueHigh = 0.0f;
            for (unsigned int pi = 0; pi < 3; ++pi) {
                for (unsigned int pj = 0; pj < 3; ++pj) {
                    valueHigh += imageHigh.getElement(i * 3 + pi, j * 3 + pj);
                }
            }
            float diff = valueLow - valueHigh / 9.0f;
            for (unsigned int pi = 0; pi < 3; ++pi) {
                for (unsigned int pj = 0; pj < 3; ++pj) {
                    imageHigh.setElement(i * 3 + pi, j * 3 + pj,
                        imageHigh.getElement(i * 3 + pi, j * 3 + pj) + diff);
                }
            }
        }
    }
}

__global__
void _globalOptimize(float* image_high, const float* image_low, unsigned int m, unsigned int n) {
    unsigned int global_i = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int global_j = threadIdx.x + blockIdx.x * blockDim.x;

    if (global_i < m && global_j < n) {
        float value_low = image_low[global_i * n + global_j];
        float value_high = 0.0f;
        for (unsigned int i = 0; i < 3; ++i) {
            for (unsigned int j = 0; j < 3; ++j) {
                value_high += image_high[(global_i * 3 + i) * (n * 3) + (global_j * 3 + j)];
            }
        }
        float diff = value_low - value_high / 9.0f;
        for (unsigned int i = 0; i < 3; ++i) {
            for (unsigned int j = 0; j < 3; ++j) {
                image_high[(global_i * 3 + i) * (n * 3) + (global_j * 3 + j)] += diff;
            }
        }
    }
}

template <>
void SuperResolution<DeviceMatrix>::globalOptimize(DeviceMatrix& imageHigh, const DeviceMatrix& imageLow) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((imageLow.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (imageLow.getHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    _globalOptimize<<<gridDim, blockDim>>>(imageHigh.getElements(), imageLow.getElements(),
        imageLow.getHeight(), imageLow.getWidth());
}