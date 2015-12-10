#include "MatrixOperations.cuh"

template <>
HostMatrix MatrixOperations<HostMatrix>::deepCopy(const HostMatrix& matrix) {
    HostMatrix matrixCopy(matrix.getHeight(), matrix.getWidth());
    size_t count = matrix.getHeight() * matrix.getWidth() * sizeof(float);
    cudaMemcpy(matrixCopy.getElements(), matrix.getElements(), count, cudaMemcpyHostToHost);
    return matrixCopy;
}

template <>
DeviceMatrix MatrixOperations<DeviceMatrix>::deepCopy(const DeviceMatrix& matrix) {
    DeviceMatrix matrixCopy(matrix.getHeight(), matrix.getWidth());
    size_t count = matrix.getHeight() * matrix.getWidth() * sizeof(float);
    cudaMemcpy(matrixCopy.getElements(), matrix.getElements(), count, cudaMemcpyDeviceToDevice);
    return matrixCopy;
}

// template <>
// HostMatrix MatrixOperations<HostMatrix>::pad(const HostMatrix& matrix,
//     unsigned int height, unsigned int width, unsigned int i, unsigned int j, float value) {
//     if (height < matrix.getHeight() || width < matrix.getWidth()) {
//         throw std::invalid_argument("Invalid argument.");
//     } else {
//         HostMatrix matrixPadded(height, width);
//         unsigned int ii = i + matrix.getHeight();
//         unsigned int jj = j + matrix.getWidth();
//         for (unsigned int pi = 0; pi < height; ++pi) {
//             for (unsigned int pj = 0; pj < width; ++pj) {
//                 if (pi >= i && pi < ii && pj >= j && pj < jj) {
//                     matrixPadded.setElement(pi, pj, matrix.getElement(pi - i, pj - j));
//                 } else {
//                     matrixPadded.setElement(pi, pj, value);
//                 }
//             }
//         }
//         return matrixPadded;
//     }
// }
//
// __global__
// void _pad(const float* matrix_in, float* matrix_out, unsigned int m_in, unsigned int n_in,
//     unsigned int m_out, unsigned int n_out, unsigned int i, unsigned int j, float value) {
//     unsigned int global_i = threadIdx.y + blockIdx.y * blockDim.y;
//     unsigned int global_j = threadIdx.x + blockIdx.x * blockDim.x;
//
//     if (global_i < m_out && global_j < n_out) {
//         int global_i_in = global_i - i;
//         int global_j_in = global_j - j;
//         matrix_out[global_i * n_out + global_j] =
//             (global_i_in >= 0 && global_i_in < m_in && global_j_in >= 0 && global_j_in < n_in)?
//             matrix_in[global_i_in * n_in + global_j_in]: value;
//     }
// }
//
// template <>
// DeviceMatrix MatrixOperations<DeviceMatrix>::pad(const DeviceMatrix& matrix,
//     unsigned int height, unsigned int width, unsigned int i, unsigned int j, float value) {
//     if (height < matrix.getHeight() || width < matrix.getWidth()) {
//         throw std::invalid_argument("Invalid argument.");
//     } else {
//         const unsigned int BLOCK_SIZE = 32;
//
//         DeviceMatrix matrixPadded(height, width);
//         dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
//         dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
//         _pad<<<gridDim, blockDim>>>(matrix.getElements(), matrixPadded.getElements(),
//             matrix.getHeight(), matrix.getWidth(), height, width, i, j, value);
//         return matrixPadded;
//     }
// }

template <>
HostMatrix MatrixOperations<HostMatrix>::transpose(const HostMatrix& matrix) {
    HostMatrix matrixTranspose(matrix.getWidth(), matrix.getHeight());
    for (unsigned int i = 0; i < matrixTranspose.getHeight(); ++i) {
        for (unsigned int j = 0; j < matrixTranspose.getWidth(); ++j) {
            matrixTranspose.setElement(i, j, matrix.getElement(j, i));
        }
    }
    return matrixTranspose;
}

__global__
void _transpose(const float* matrix_in, float* matrix_out, unsigned int m, unsigned int n) {
    unsigned int global_i = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int global_j = threadIdx.x + blockIdx.x * blockDim.x;

    if (global_i < m && global_j < n) {
        matrix_out[global_i * n + global_j] = matrix_in[global_j * m + global_i];
    }
}

template <>
DeviceMatrix MatrixOperations<DeviceMatrix>::transpose(const DeviceMatrix& matrix) {
    const unsigned int BLOCK_SIZE = 32;

    DeviceMatrix matrixTranspose(matrix.getWidth(), matrix.getHeight());
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((matrixTranspose.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (matrixTranspose.getHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    _transpose<<<gridDim, blockDim>>>(matrix.getElements(), matrixTranspose.getElements(),
        matrixTranspose.getHeight(), matrixTranspose.getWidth());
    return matrixTranspose;
}

template <>
HostMatrix MatrixOperations<HostMatrix>::concatenateRows(
    const HostMatrix& matrixUpper, const HostMatrix& matrixLower) {
    if (matrixUpper.getWidth() != matrixLower.getWidth()) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        HostMatrix matrixConcat(matrixUpper.getHeight() + matrixLower.getHeight(), matrixUpper.getWidth());
        for (unsigned int i = 0; i < matrixUpper.getHeight(); ++i) {
            for (unsigned int j = 0; j < matrixConcat.getWidth(); ++j) {
                matrixConcat.setElement(i, j, matrixUpper.getElement(i, j));
            }
        }
        unsigned int offset = matrixUpper.getHeight();
        for (unsigned int i = 0; i < matrixLower.getHeight(); ++i) {
            for (unsigned int j = 0; j < matrixConcat.getWidth(); ++j) {
                matrixConcat.setElement(i + offset, j, matrixLower.getElement(i, j));
            }
        }
        return matrixConcat;
    }
}

__global__
void _concatenateRows(const float* matrix_upper, const float* matrix_lower, float* matrix_out,
    unsigned int m_upper, unsigned int m_lower, unsigned int n) {
    unsigned int global_i = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int global_j = threadIdx.x + blockIdx.x * blockDim.x;

    if (global_j < n) {
        if (global_i < m_upper) {
            matrix_out[global_i * n + global_j] = matrix_upper[global_i * n + global_j];
        } else if (global_i < m_upper + m_lower) {
            matrix_out[global_i * n + global_j] = matrix_lower[(global_i - m_upper) * n + global_j];
        }
    }
}

template <>
DeviceMatrix MatrixOperations<DeviceMatrix>::concatenateRows(
    const DeviceMatrix& matrixUpper, const DeviceMatrix& matrixLower) {
    if (matrixUpper.getWidth() != matrixLower.getWidth()) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        const unsigned int BLOCK_SIZE = 32;

        DeviceMatrix matrixConcat(matrixUpper.getHeight() + matrixLower.getHeight(), matrixUpper.getWidth());
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((matrixConcat.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (matrixConcat.getHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE);
        _concatenateRows<<<gridDim, blockDim>>>(matrixUpper.getElements(), matrixLower.getElements(),
            matrixConcat.getElements(), matrixUpper.getHeight(), matrixLower.getHeight(), matrixConcat.getWidth());
        return matrixConcat;
    }
}