#pragma once

#include "utils.hpp"
class imageRegistrator
{
private:
    const int _height;
    const int _width;
    const int _size;
    const float _heightd;
    const float _widthd;
    const float logbase;
    const std::vector<std::pair<float,float>> map;
    const std::vector<float> filter;
    const std::vector<std::pair<int,float>> mask;
    FixedQueue<std::vector<std::complex<float>>, 2> queue;

    std::vector<std::pair<float,float>> getPolarMap();
    std::vector<std::pair<int,float>> apodizeMask();
    std::vector<float> gaussianHPF (float sigma);
    std::vector<float> mapCoordinates(std::vector<float> &img);
    void mapCoordinates(std::vector<float> &img, std::vector<float> &output);
    void mapCoordinates(std::vector<float> &img, std::vector<std::complex<float>> &output);
    void centerOfMass(const std::vector<std::complex<float>> &img, int m, std::pair<float, float> &com);

    fftwf_plan fft_forward;
    fftwf_plan fft_backward;

public:
    imageRegistrator(int height, int width);
    ~imageRegistrator();

    float getHeight();
    float getWidth();


    void fftShift(std::vector<std::complex<float>> &img, const bool forward = true);
    template <typename T>
    void apodize (std::vector<T> &img);
    void phaseCorrelation(std::vector<std::complex<float>> img1,std::vector<std::complex<float>> img2,std::pair<float,float> &results);

    template <typename T>
    void logPolarTransform(std::vector<std::complex<float>> &img, std::vector<T> &output);
    std::pair<float, float> getAngScale(std::vector<std::complex<float>> &img1,std::vector<std::complex<float>> &img2);

    std::vector<float> total_t;
    std::vector<float> total_t1;
    std::chrono::duration<float, std::milli> ms_float;
    
    void append(std::vector<std::complex<float>> &img);
    std::pair<float,float> getAngScale();
};
