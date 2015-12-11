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
HostMatrix MatrixOperations<HostMatrix>::padRowsTop(const HostMatrix& matrix,
    unsigned int rows, float value) {
    HostMatrix matrixPadded(rows + matrix.getHeight(), matrix.getWidth());
    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < matrixPadded.getWidth(); ++j) {
            matrixPadded.setElement(i, j, value);
        }
    }
    for (unsigned int i = 0; i < matrix.getHeight(); ++i) {
        for (unsigned int j = 0; j < matrixPadded.getWidth(); ++j) {
            matrixPadded.setElement(rows + i, j, matrix.getElement(i, j));
        }
    }
    return matrixPadded;
}

__global__
void _padRowsTop(const float* matrix_in, float* matrix_out, unsigned int m, unsigned int n,
    unsigned int rows, float value) {
    unsigned int global_i = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int global_j = threadIdx.x + blockIdx.x * blockDim.x;

    if (global_i < m && global_j < n) {
        matrix_out[global_i * n + global_j] =
            global_i >= rows? matrix_in[(global_i - rows) * n + global_j]: value;
    }
}

template <>
DeviceMatrix MatrixOperations<DeviceMatrix>::padRowsTop(const DeviceMatrix& matrix,
    unsigned int rows, float value) {
    const unsigned int BLOCK_SIZE = 32;

    DeviceMatrix matrixPadded(rows + matrix.getHeight(), matrix.getWidth());
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((matrixPadded.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (matrixPadded.getHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    _padRowsTop<<<gridDim, blockDim>>>(matrix.getElements(), matrixPadded.getElements(),
        matrixPadded.getHeight(), matrixPadded.getWidth(), rows, value);
    return matrixPadded;
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

template <>
HostMatrix MatrixOperations<HostMatrix>::index(const HostMatrix& matrix, const HostMatrix& vector) {
    HostMatrix matrixIndexed(matrix.getHeight(), vector.getWidth());
    for (unsigned int j = 0; j < matrixIndexed.getWidth(); ++j) {
        unsigned int indexJ = (unsigned int)roundf(vector.getElement(0, j));
        for (unsigned int i = 0; i < matrixIndexed.getHeight(); ++i) {
            matrixIndexed.setElement(i, j, matrix.getElement(i, indexJ));            
        }
    }
    return matrixIndexed;
}

__global__
void _index(const float* matrix_in, const float* vector, float* matrix_out,
    unsigned int m, unsigned int p, unsigned int n) {
    unsigned int global_i = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int global_j = threadIdx.x + blockIdx.x * blockDim.x;

    if (global_i < m && global_j < n) {
        unsigned int indexJ = (unsigned int)roundf(vector[global_j]);
        matrix_out[global_i * n + global_j] = matrix_in[global_i * p + indexJ];
    }
}

template <>
DeviceMatrix MatrixOperations<DeviceMatrix>::index(const DeviceMatrix& matrix, const DeviceMatrix& vector) {
    const unsigned int BLOCK_SIZE = 32;

    DeviceMatrix matrixIndexed(matrix.getHeight(), vector.getWidth());
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((matrixIndexed.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (matrixIndexed.getHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    _index<<<gridDim, blockDim>>>(matrix.getElements(), vector.getElements(), matrixIndexed.getElements(),
        matrix.getHeight(), matrix.getWidth(), vector.getWidth());
    return matrixIndexed;
}