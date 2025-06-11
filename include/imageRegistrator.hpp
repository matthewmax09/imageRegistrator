#pragma once

class imageRegistrator
{
private:
    const int _height;
    const int _width;
    const int _size;
    const double _heightd;
    const double _widthd;
    const std::vector<std::pair<double,double>> map;
    const std::vector<double> filter;
    

    std::vector<std::pair<double,double>> getPolarMap();
    std::vector<double> gaussianHPF (double sigma);
    std::vector<double> mapCoordinates(std::vector<double> &img);
    void centerOfMass(const std::vector<std::complex<double>> &img, int m, std::pair<double, double> &com);

    fftw_plan fft_forward;
    fftw_plan fft_backward;

public:
    imageRegistrator(int height, int width);
    ~imageRegistrator();

    double getHeight();
    double getWidth();


    void fftShift(std::vector<std::complex<double>> &img, const bool forward = true);
    void apodize (std::vector<std::complex<double>> &img);
    void phaseCorrelation(std::vector<std::complex<double>> &img1,std::vector<std::complex<double>> &img2);
    std::vector<double> logPolarTransform(std::vector<std::complex<double>> &img);

};
