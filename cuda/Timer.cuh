#ifndef _TIMER_CUH_
#define _TIMER_CUH_

class Timer {
public:
    Timer();
    ~Timer();

    void start();
    float stop();

private:
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
};

#endif