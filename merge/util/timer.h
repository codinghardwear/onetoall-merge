#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer 
{
    std::chrono::steady_clock::time_point time_begin;
 
public:
    Timer() 
    {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeSeconds() 
    {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return std::chrono::duration<float>(time_end - time_begin).count();
    }

    void reset() 
    {
        time_begin = std::chrono::steady_clock::now();
    }
};

#endif