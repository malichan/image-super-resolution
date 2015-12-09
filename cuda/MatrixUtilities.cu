#include "MatrixUtilities.cuh"

HostMatrix MatrixUtilities::loadFromFile(const char* fileName) {
    std::ifstream fin;
    fin.open(fileName);

    unsigned int height = 0;
    unsigned int width = 0;
    fin >> height >> width;

    HostMatrix matrix(height, width);
    for (unsigned int i = 0; i < height; ++i) {
        for (unsigned int j = 0; j < width; ++j) {
            float value = 0.0f;
            fin >> value;
            matrix.setElement(i, j, value);
        }
    }

    fin.close();
    return matrix;
}

void MatrixUtilities::saveToFile(const HostMatrix& matrix, const char* fileName) {
    std::ofstream fout;
    fout.open(fileName);

    fout.setf(std::ios::fixed, std::ios::floatfield);
    fout.precision(6);
    fout << matrix.getHeight() << " " << matrix.getWidth() << std::endl;
    for (unsigned int i = 0; i < matrix.getHeight(); ++i) {
        for (unsigned int j = 0; j < matrix.getWidth(); ++j) {
            fout << matrix.getElement(i, j) << " ";
        }
        fout << std::endl;
    }

    fout.close();
}

bool MatrixUtilities::compare(const HostMatrix& matrixA, const HostMatrix& matrixB, float epsilon) {
    if (matrixA.getHeight() != matrixB.getHeight() || matrixA.getWidth() != matrixB.getWidth()) {
        return false;
    } else {
        for (unsigned int i = 0; i < matrixA.getHeight(); ++i) {
            for (unsigned int j = 0; j < matrixA.getWidth(); ++j) {
                if (fabs(matrixA.getElement(i, j) - matrixB.getElement(i, j)) > epsilon) {
                    return false;
                }
            }
        }
        return true;
    }
}

HostMatrix MatrixUtilities::copyToHost(const Matrix& matrix) {
    HostMatrix matrixCopy(matrix.getHeight(), matrix.getWidth());
    size_t count = matrix.getHeight() * matrix.getWidth() * sizeof(float);
    cudaMemcpyKind kind = matrix.isOnDevice()? cudaMemcpyDeviceToHost: cudaMemcpyHostToHost;
    cudaMemcpy(matrixCopy.getElements(), matrix.getElements(), count, kind);
    return matrixCopy;
}

DeviceMatrix MatrixUtilities::copyToDevice(const Matrix& matrix) {
    DeviceMatrix matrixCopy(matrix.getHeight(), matrix.getWidth());
    size_t count = matrix.getHeight() * matrix.getWidth() * sizeof(float);
    cudaMemcpyKind kind = matrix.isOnDevice()? cudaMemcpyDeviceToDevice: cudaMemcpyHostToDevice;
    cudaMemcpy(matrixCopy.getElements(), matrix.getElements(), count, kind);
    return matrixCopy;
}