#include "Bitmap.cuh"

HostMatrix Bitmap::loadFromFile(const char* fileName) {
    std::ifstream fin;
    fin.open(fileName);
    if (!fin.is_open()) {
        std::cerr << "Could not open " << fileName << "." << std::endl;
        exit(EXIT_FAILURE);
    }

    unsigned int buffer = 0;

    // BMP Header
    buffer = 0;             fin.read((char*)&buffer, 2); // BM
    if (buffer != 19778) {
        std::cerr << "The image format is not supported. " <<
            "(Offse = " << fin.tellg() << ", Value = " << buffer << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    fin.seekg(12, std::ios::cur);

    // DIB Header
    buffer = 0;             fin.read((char*)&buffer, 4); // DIB Header Size
    if (buffer != 40) {
        std::cerr << "The image format is not supported. " <<
            "(Offse = " << fin.tellg() << ", Value = " << buffer << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    buffer = 0;             fin.read((char*)&buffer, 4); // Width
    unsigned int width = buffer;
    buffer = 0;             fin.read((char*)&buffer, 4); // Height
    unsigned int height = buffer;
    fin.seekg(2, std::ios::cur);
    buffer = 0;             fin.read((char*)&buffer, 2); // Number of Bits per Pixel
    if (buffer != 8 && buffer != 24) {
        std::cerr << "The image format is not supported. " <<
            "(Offse = " << fin.tellg() << ", Value = " << buffer << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    bool grayScale = (buffer == 8);
    buffer = 0;             fin.read((char*)&buffer, 4); // Compression
    if (buffer != 0) {
        std::cerr << "The image format is not supported. " <<
            "(Offse = " << fin.tellg() << ", Value = " << buffer << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    fin.seekg(20, std::ios::cur);

    // Color Palette
    if (grayScale) {
        fin.seekg(1024, std::ios::cur);
    }

    // Image Data
    HostMatrix image(height, width);
    if (grayScale) {
        unsigned int padding = (width + 3) / 4 * 4 - width;
        for (unsigned int i = 0; i < height; ++i) {
            for (unsigned int j = 0; j < width; ++j) {
                buffer = 0; fin.read((char*)&buffer, 1);
                image.setElement(i, j, convertToFloat(buffer));
            }
            fin.seekg(padding, std::ios::cur);
        }
    } else {
        unsigned int padding = (3 * width + 3) / 4 * 4 - width * 3;
        for (unsigned int i = 0; i < height; ++i) {
            for (unsigned int j = 0; j < width; ++j) {
                float value = 0.0f;
                buffer = 0; fin.read((char*)&buffer, 1);
                value += convertToFloat(buffer) * 0.11f;
                buffer = 0; fin.read((char*)&buffer, 1);
                value += convertToFloat(buffer) * 0.59f;
                buffer = 0; fin.read((char*)&buffer, 1);
                value += convertToFloat(buffer) * 0.30f;
                image.setElement(i, j, value);
            }
            fin.seekg(padding, std::ios::cur);
        }
    }

    fin.close();
    return image;
}

void Bitmap::saveToFile(const HostMatrix& image, const char* fileName) {
    std::ofstream fout;
    fout.open(fileName);
    if (!fout.is_open()) {
        std::cerr << "Could not open " << fileName << "." << std::endl;
        exit(EXIT_FAILURE);
    }

    unsigned int height = image.getHeight();
    unsigned int width = image.getWidth();
    unsigned int padding = (width + 3) / 4 * 4 - width;
    unsigned int headerSize = 14 + 40 + 1024;
    unsigned int imageSize = height * (width + padding);
    unsigned int fileSize = headerSize + imageSize;

    unsigned int buffer = 0;

    // BMP Header
    buffer = 19778;         fout.write((char*)&buffer, 2);  // BM
    buffer = fileSize;      fout.write((char*)&buffer, 4);  // File Size
    buffer = 0;             fout.write((char*)&buffer, 2);  // Reserved
    buffer = 0;             fout.write((char*)&buffer, 2);  // Reserved
    buffer = headerSize;    fout.write((char*)&buffer, 4);  // Offset

    // DIB Header
    buffer = 40;            fout.write((char*)&buffer, 4);  // DIB Header Size
    buffer = width;         fout.write((char*)&buffer, 4);  // Width
    buffer = height;        fout.write((char*)&buffer, 4);  // Height
    buffer = 1;             fout.write((char*)&buffer, 2);  // Number of Color Planes
    buffer = 8;             fout.write((char*)&buffer, 2);  // Number of Bits per Pixel
    buffer = 0;             fout.write((char*)&buffer, 4);  // Compression
    buffer = imageSize;     fout.write((char*)&buffer, 4);  // Image Data Size
    buffer = 0;             fout.write((char*)&buffer, 4);  // Horizontal Resolution
    buffer = 0;             fout.write((char*)&buffer, 4);  // Vertical Resolution
    buffer = 0;             fout.write((char*)&buffer, 4);  // Number of Colors in Color Palette
    buffer = 0;             fout.write((char*)&buffer, 4);  // Number of Important Colors

    // Color Palette
    for (unsigned int shade = 0; shade < 256; ++shade) {
        buffer = shade;     fout.write((char*)&buffer, 1);  // Red
                            fout.write((char*)&buffer, 1);  // Green
                            fout.write((char*)&buffer, 1);  // Blue
        buffer = 0;         fout.write((char*)&buffer, 1);  // Padding
    }

    // Image Data
    for (unsigned int i = 0; i < height; ++i) {
        for (unsigned int j = 0; j < width; ++j) {
            buffer = convertToByte(image.getElement(i, j));
            fout.write((char*)&buffer, 1);
        }
        buffer = 0;         fout.write((char*)&buffer, padding);  // Padding
    }

    fout.close();
}