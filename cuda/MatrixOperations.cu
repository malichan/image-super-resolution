#include "MatrixOperations.cuh"

class Fill {
public:
    float value;

    Fill(float value): value(value) {}

    __host__ __device__
    float operator()(float in) const {
        return value;
    }
};

template <>
void MatrixOperations<HostMatrix>::fill(HostMatrix& matrix, float value) {
    transform(matrix, Fill(value));
}

template <>
void MatrixOperations<DeviceMatrix>::fill(DeviceMatrix& matrix, float value) {
    transform(matrix, Fill(value));
}

template <>
HostMatrix MatrixOperations<HostMatrix>::multiply(const HostMatrix& matrixA, const HostMatrix& matrixB) {
    if (matrixA.getWidth() != matrixB.getHeight()) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        HostMatrix matrixC(matrixA.getHeight(), matrixB.getWidth());
        for (unsigned int i = 0; i < matrixC.getHeight(); ++i) {
            for (unsigned int j = 0; j < matrixC.getWidth(); ++j) {
                float value = 0.0f;
                for (unsigned int k = 0; k < matrixA.getWidth(); ++k) {
                    value += matrixA.getElement(i, k) * matrixB.getElement(k, j);
                }
                matrixC.setElement(i, j, value);
            }
        }
        return matrixC;
    }
}

__global__
void _multiply(const float* matrix_a, const float* matrix_b, float* matrix_c,
    unsigned int m, unsigned int p, unsigned int n) {
    __shared__ extern float shared_mem[];
    float* s_a = shared_mem;
    float* s_b = shared_mem + blockDim.x * blockDim.y;

    unsigned int local_i = threadIdx.y;
    unsigned int local_j = threadIdx.x;
    unsigned int global_i = 0;
    unsigned int global_j = 0;

    unsigned int offset_i_a = blockIdx.y * blockDim.y;
    unsigned int offset_j_a = 0;
    unsigned int step_j_a = blockDim.x;
    unsigned int offset_i_b = 0;
    unsigned int offset_j_b = blockIdx.x * blockDim.x;
    unsigned int step_i_b = blockDim.y;

    unsigned int loop = (p + blockDim.x - 1) / blockDim.x;
    float value = 0.0f;
    for (unsigned int l = 0; l < loop; ++l, offset_j_a += step_j_a, offset_i_b += step_i_b) {
        global_i = offset_i_a + local_i;
        global_j = offset_j_a + local_j;
        s_a[local_i * blockDim.x + local_j] = (global_i < m && global_j < p)?
            matrix_a[global_i * p + global_j]: 0.0f;
        global_i = offset_i_b + local_i;
        global_j = offset_j_b + local_j;
        s_b[local_i * blockDim.x + local_j] = (global_i < p && global_j < n)?
            matrix_b[global_i * n + global_j]: 0.0f;
        __syncthreads();

        for (unsigned int k = 0; k < blockDim.x; ++k) {
            value += s_a[local_i * blockDim.x + k] * s_b[k * blockDim.x + local_j];
        }
        __syncthreads();
    }

    global_i = threadIdx.y + blockIdx.y * blockDim.y;
    global_j = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_i < m && global_j < n) {
        matrix_c[global_i * n + global_j] = value;
    }
}

template <>
DeviceMatrix MatrixOperations<DeviceMatrix>::multiply(const DeviceMatrix& matrixA, const DeviceMatrix& matrixB) {
    if (matrixA.getWidth() != matrixB.getHeight()) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        const unsigned int BLOCK_SIZE = 32;

        DeviceMatrix matrixC(matrixA.getHeight(), matrixB.getWidth());
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((matrixC.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (matrixC.getHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE);
        unsigned int sharedMem = BLOCK_SIZE * BLOCK_SIZE * sizeof(float) * 2;
        _multiply<<<gridDim, blockDim, sharedMem>>>(matrixA.getElements(), matrixB.getElements(), matrixC.getElements(),
            matrixC.getHeight(), matrixA.getWidth(), matrixC.getWidth());

        return matrixC;
    }
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
HostMatrix MatrixOperations<HostMatrix>::concatenateVertical(
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
void _concatenateVertical(const float* matrix_upper, const float* matrix_lower, float* matrix_out,
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
DeviceMatrix MatrixOperations<DeviceMatrix>::concatenateVertical(
    const DeviceMatrix& matrixUpper, const DeviceMatrix& matrixLower) {
    if (matrixUpper.getWidth() != matrixLower.getWidth()) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        const unsigned int BLOCK_SIZE = 32;

        DeviceMatrix matrixConcat(matrixUpper.getHeight() + matrixLower.getHeight(), matrixUpper.getWidth());
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((matrixConcat.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (matrixConcat.getHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE);
        _concatenateVertical<<<gridDim, blockDim>>>(matrixUpper.getElements(), matrixLower.getElements(),
            matrixConcat.getElements(), matrixUpper.getHeight(), matrixLower.getHeight(), matrixConcat.getWidth());
        return matrixConcat;
    }
}