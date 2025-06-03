#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <glog/logging.h>

// General Array utils

// https://stackoverflow.com/questions/28548703/why-does-stdnth-element-return-sorted-vectors-for-input-vectors-with-n-33-el
template <typename T>
T percentile(std::vector<T> vectorIn, int percent)
{
    CHECK_LE(percent,100);
    CHECK_GE(percent,0);
    
    auto nth = vectorIn.begin() + (percent*vectorIn.size())/100;
    std::nth_element(vectorIn.begin(), nth, vectorIn.end());
    return *nth;
}

// https://gist.github.com/lorenzoriano/5414671
template <typename T>
std::vector<T> linspace(double start, double end, double num, bool endpoint=true)
{
    CHECK_NE(num,0);
    std::vector<T> linspaced;
    
    // Pushback exact start
    linspaced.push_back(static_cast<T>(start));
    
    if (num == 1) {return linspaced;}
    if (endpoint) --num;
    double delta = (end - start)/num;
    for (int i = 1; i < num; i++)
    {
        linspaced.push_back(static_cast<T>(start + delta*i));
    }
    
    // if include endpoint, pushback exact end
    if (endpoint) linspaced.push_back(static_cast<T>(end));
    
    return linspaced;
}

// Copied and modified from cv::getGaussianKernelBitExact
double getGaussianKernel(std::vector<double>& result, int n, double sigma);

std::vector<double> hanning_window(int window_size);

template<typename T>
T inline roundCast(double a){
    return static_cast<T>(std::round(a));
}