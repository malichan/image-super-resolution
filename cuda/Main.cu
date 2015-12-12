#include "SuperResolution.cuh"
#include "Bitmap.cuh"
#include "Timer.cuh"

void showHelp() {
    std::cout << "Usage: ./sr3x <input-image> <output-image> <c|g>" << std::endl;
    std::cout << "The input image must be a BMP file (8-bit grayscale or 24-bit RGB)." << std::endl;
    std::cout << "The output image will be a BMP file (8-bit grayscale)." << std::endl;
    std::cout << "The image will be processed on CPU with option 'c', or on GPU with 'g'." << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        showHelp();
        return 0;
    } else {
        char* inputFileName = argv[1];
        char* outputFileName = argv[2];
        char option = argv[3][0];

        if (option == 'c') {
            std::cout << "Scaling up the image on CPU ... " << std::endl;

            Timer timer;
            timer.start();

            HostMatrix inputImage = Bitmap::loadFromFile(inputFileName);
            SuperResolution<HostMatrix> sr;
            HostMatrix outputImage = sr.scaleUp3x(inputImage);
            Bitmap::saveToFile(outputImage, outputFileName);

            float timeElapsed = timer.stop();

            std::cout << "Input image size: " << inputImage.getHeight() << " x " <<
                inputImage.getWidth() << std::endl;
            std::cout << "Output image size: " << outputImage.getHeight() << " x " <<
                outputImage.getWidth() << std::endl;
            std::cout << "Processing time: " << std::fixed << std::setprecision(3) <<
                timeElapsed << " ms" << std::endl;
            return 0;
        } else if (option == 'g') {
            std::cout << "Scaling up the image on GPU ... " << std::endl;

            Timer timer;
            timer.start();

            HostMatrix inputImage = Bitmap::loadFromFile(inputFileName);
            DeviceMatrix dInputImage = MatrixUtilities::copyToDevice(inputImage);
            SuperResolution<DeviceMatrix> sr;
            DeviceMatrix dOutputImage = sr.scaleUp3x(dInputImage);
            HostMatrix outputImage = MatrixUtilities::copyToHost(dOutputImage);
            Bitmap::saveToFile(outputImage, outputFileName);

            float timeElapsed = timer.stop();

            std::cout << "Input image size: " << inputImage.getHeight() << " x " <<
                inputImage.getWidth() << std::endl;
            std::cout << "Output image size: " << outputImage.getHeight() << " x " <<
                outputImage.getWidth() << std::endl;
            std::cout << "Processing time: " << std::fixed << std::setprecision(3) <<
                timeElapsed << " ms" << std::endl;
            return 0;
        } else {
            std::cerr << "Invalid option." << std::endl;
            showHelp();
            exit(EXIT_FAILURE);
        }
    }
}