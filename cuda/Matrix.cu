#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "Matrix.cuh"

inline HostMatrix::HostMatrix(unsigned int height, unsigned int width): Matrix(height, width, false) {
    cudaMallocHost(&elements, height * width * sizeof(float));
}

inline HostMatrix::~HostMatrix() {
    cudaFreeHost(elements);
}

inline float HostMatrix::getElement(unsigned int i, unsigned int j) const {
    if (i >= height || j >= width) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        return elements[i * width + j];
    }
}

inline void HostMatrix::setElement(unsigned int i, unsigned int j, float value) {
    if (i >= height || j >= width) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        elements[i * width + j] = value;
    }
}

inline DeviceMatrix::DeviceMatrix(unsigned int height, unsigned int width): Matrix(height, width, true) {
    cudaMalloc(&elements, height * width * sizeof(float));
}

inline DeviceMatrix::~DeviceMatrix() {
    cudaFree(elements);
}

inline float DeviceMatrix::getElement(unsigned int i, unsigned int j) const {
    if (i >= height || j >= width) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        float value = 0.0f;
        cudaMemcpy(&value, &elements[i * width + j], sizeof(float), cudaMemcpyDeviceToHost);
        return value;
    }
}

inline void DeviceMatrix::setElement(unsigned int i, unsigned int j, float value) {
    if (i >= height || j >= width) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        cudaMemcpy(&elements[i * width + j], &value, sizeof(float), cudaMemcpyHostToDevice);
    }
}

HostMatrix MatrixUtilities::loadFromFile(const char* fileName) {
    std::ifstream fin;
    fin.open(fileName);

    unsigned int height = 0;
    unsigned int width = 0;
    fin >> height >> width;

    HostMatrix matrix(height, width);
    for (unsigned int k = 0; k < height * width; ++k) {
        fin >> matrix.elements[k];
    }

    fin.close();
    return matrix;
}

void MatrixUtilities::saveToFile(const HostMatrix& matrix, const char* fileName) {
    std::ofstream fout;
    fout.open(fileName);

    fout.setf(std::ios::fixed, std::ios::floatfield);
    fout.precision(6);
    fout << matrix.height << " " << matrix.width << std::endl;
    for (unsigned int i = 0; i < matrix.height; ++i) {
        for (unsigned int j = 0; j < matrix.width; ++j) {
            fout << matrix.getElement(i, j) << " ";
        }
        fout << std::endl;
    }

    fout.close();
}

bool MatrixUtilities::compare(const HostMatrix& matrixA, const HostMatrix& matrixB, float epsilon) {
    if (matrixA.height != matrixB.height || matrixA.width != matrixB.width) {
        return false;
    } else {
        for (unsigned int k = 0; k < matrixA.height * matrixA.width; ++k) {
            if (fabs(matrixA.elements[k] - matrixB.elements[k]) > epsilon) {
                return false;
            }
        }
        return true;
    }
}

HostMatrix MatrixUtilities::copyToHost(const Matrix& matrix) {
    HostMatrix matrixCopy(matrix.height, matrix.width);
    size_t count = matrix.height * matrix.width * sizeof(float);
    cudaMemcpyKind kind = matrix.onDevice? cudaMemcpyDeviceToHost: cudaMemcpyHostToHost;
    cudaMemcpy(matrixCopy.elements, matrix.elements, count, kind);
    return matrixCopy;
}

DeviceMatrix MatrixUtilities::copyToDevice(const Matrix& matrix) {
    DeviceMatrix matrixCopy(matrix.height, matrix.width);
    size_t count = matrix.height * matrix.width * sizeof(float);
    cudaMemcpyKind kind = matrix.onDevice? cudaMemcpyDeviceToDevice: cudaMemcpyHostToDevice;
    cudaMemcpy(matrixCopy.elements, matrix.elements, count, kind);
    return matrixCopy;
}

HostMatrix MatrixUtilities::multiplyOnHost(const HostMatrix& matrixA, const HostMatrix& matrixB) {
    if (matrixA.width != matrixB.height) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        HostMatrix matrixC(matrixA.height, matrixB.width);
        for (unsigned int i = 0; i < matrixC.height; ++i) {
            for (unsigned int j = 0; j < matrixC.width; ++j) {
                float value = 0.0f;
                for (unsigned int k = 0; k < matrixA.width; ++k) {
                    value += matrixA.getElement(i, k) * matrixB.getElement(k, j);
                }
                matrixC.setElement(i, j, value);
            }
        }
        return matrixC;
    }
}

__global__
void _multiply(float* matrix_a, float* matrix_b, float* matrix_c, unsigned int m, unsigned int p, unsigned int n) {
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

DeviceMatrix MatrixUtilities::multiplyOnDevice(const DeviceMatrix& matrixA, const DeviceMatrix& matrixB) {
    if (matrixA.width != matrixB.height) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        const unsigned int BLOCK_SIZE = 32;

        DeviceMatrix matrixC(matrixA.height, matrixB.width);
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((matrixC.width + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (matrixC.height + BLOCK_SIZE - 1) / BLOCK_SIZE);
        unsigned int sharedMem = BLOCK_SIZE * BLOCK_SIZE * sizeof(float) * 2;
        _multiply<<<gridDim, blockDim, sharedMem>>>(matrixA.elements, matrixB.elements, matrixC.elements,
            matrixC.height, matrixA.width, matrixC.width);

        return matrixC;
    }
}

HostMatrix MatrixUtilities::padOnHost(const HostMatrix& matrix, unsigned int height, unsigned int width,
    unsigned int i, unsigned int j, float value) {
    if (height < matrix.height || width < matrix.width) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        HostMatrix matrixPadded(height, width);
        unsigned int ii = i + matrix.height;
        unsigned int jj = j + matrix.width;
        for (unsigned int pi = 0; pi < height; ++pi) {
            for (unsigned int pj = 0; pj < width; ++pj) {
                if (pi >= i && pi < ii && pj >= j && pj < jj) {
                    matrixPadded.setElement(pi, pj, matrix.getElement(pi - i, pj - j));
                } else {
                    matrixPadded.setElement(pi, pj, value);
                }
            }
        }
        return matrixPadded;
    }
}

__global__
void _pad(float* matrix_in, float* matrix_out, unsigned int m_in, unsigned int n_in,
    unsigned int m_out, unsigned int n_out, unsigned int i, unsigned int j, float value) {
    unsigned int global_i = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int global_j = threadIdx.x + blockIdx.x * blockDim.x;

    if (global_i < m_out && global_j < n_out) {
        int global_i_in = global_i - i;
        int global_j_in = global_j - j;
        matrix_out[global_i * n_out + global_j] =
            (global_i_in >= 0 && global_i_in < m_in && global_j_in >= 0 && global_j_in < n_in)?
            matrix_in[global_i_in * n_in + global_j_in]: value;
    }
}

DeviceMatrix MatrixUtilities::padOnDevice(const DeviceMatrix& matrix, unsigned int height, unsigned int width,
    unsigned int i, unsigned int j, float value) {
    if (height < matrix.height || width < matrix.width) {
        throw std::invalid_argument("Invalid argument.");
    } else {
        const unsigned int BLOCK_SIZE = 32;

        DeviceMatrix matrixPadded(height, width);
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
        _pad<<<gridDim, blockDim>>>(matrix.elements, matrixPadded.elements, matrix.height, matrix.width,
            height, width, i, j, value);
        return matrixPadded;
    }
}