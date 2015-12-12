#include "Timer.cuh"

Timer::Timer(): startEvent(), stopEvent() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
}

Timer::~Timer() {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void Timer::start() {
    cudaEventRecord(startEvent);
    cudaEventSynchronize(startEvent);
}

float Timer::stop() {
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float timeElapsed = 0.0f;
    cudaEventElapsedTime(&timeElapsed, startEvent, stopEvent);
    return timeElapsed;
}