#include <glog/logging.h>
// #include <cmath>
#include <complex>
#include <numeric>
#include <fftw3.h>
#include "imageRegistrator.hpp"
// #include "utils.hpp"
#include <omp.h>
using namespace std::literals;

imageRegistrator::imageRegistrator(int height, int width)
:_height(height),
 _width(width),
 _size(height*width),
 _heightd(height),
 _widthd(width),
 logbase(std::pow(0.55*_heightd,1.0/_widthd)),
 map(getPolarMap()),
 filter(gaussianHPF(22)),
 mask(apodizeMask())
{
    /* Create FFTW Plans - when deploying code, 
    use FFTW_PATIENT instead of FFTW_ESTIMATE to optimize for speed */
    fftw_init_threads();
    fftw_plan_with_nthreads(2);

    std::vector<std::complex<double>> tmp( _size );
    std::vector<double> tmp1(_size);
    fft_forward = fftw_plan_dft_2d( _height ,_width,reinterpret_cast<fftw_complex*>(tmp.data()), 
                                                    reinterpret_cast<fftw_complex*>(tmp.data()), 
                                                    FFTW_FORWARD, FFTW_PATIENT );
    fft_backward = fftw_plan_dft_2d( _height ,_width,reinterpret_cast<fftw_complex*>(tmp.data()), 
                                                    reinterpret_cast<fftw_complex*>(tmp.data()), 
                                                    FFTW_BACKWARD, FFTW_PATIENT );
}

imageRegistrator::~imageRegistrator()
{
    /* deallocate FFTW arrays and plans */
    fftw_destroy_plan( fft_forward );
    fftw_destroy_plan( fft_backward );
    fftw_cleanup_threads();

}

double imageRegistrator::getHeight()
{
    return _height;
}

double imageRegistrator::getWidth()
{
    return _width;
}

std::vector<std::pair<double,double>> imageRegistrator::getPolarMap() 
{
    // Ensure map is cleared before adding to it. 
    // map.clear();
    // double logbase = std::pow(0.55*_heightd,1.0/_widthd);
    double cx = _widthd/2.0;
    double cy = _heightd/2.0;
    double ellipse_coef = _heightd / _widthd;
    std::vector<double> log(_widthd);
    std::vector<double> logx(_widthd);
    std::vector<double> ang = linspace<double>(0,-M_PI,_heightd,false);
    std::vector<std::pair<double,double>> _map;
    for (int i = 0; i<log.size(); i++)
    {
        log[i] = std::pow(logbase,i);
        logx[i] = log[i]/ellipse_coef;
    }
    for (auto a : ang){
        for (int i = 0; i<log.size(); i++){
            // x = l*cos(a)+cx
            // y = l*sin(a)+cy
            _map.emplace_back(std::make_pair(logx[i]*cos(a)+cx,log[i]*sin(a)+cy));
        }
    }
    return _map;
}

std::vector<std::pair<int,double>> imageRegistrator::apodizeMask()
{
    std::vector<std::pair<int,double>> mask;
    //int aporad = _width *0.12;
    int apowidth = _width * 0.12;
    int apoheight = _height * 0.12;
    std::vector<double> han_width = hanning_window(apowidth*2);
    std::vector<double> han_height = hanning_window(apoheight*2);
    for (int i = 0; i < apoheight; i++) {
        int step = i*_width;
        int rstep = (_height-1-i)*_width;
        // Iterate top and bottom midsection
        for (int k = apowidth; k < _width-apowidth; k++){
            mask.emplace_back(std::make_pair(step+k,han_height[i]));
            mask.emplace_back(std::make_pair(rstep+k,han_height[i]));
        }
        // Iterate through the 4 corners at one shot (works for even width and height)
        for (int k = 0; k < apowidth; k++) {
            double tmp = han_height[i]*han_width[k];
            mask.emplace_back(std::make_pair(step+k,tmp));
            mask.emplace_back(std::make_pair(step+_width-1-k,tmp));
            mask.emplace_back(std::make_pair(rstep+k,tmp));
            mask.emplace_back(std::make_pair(rstep+_width-1-k,tmp));
        }
    }
    // Iterate left and right midsection
    for (int i = apoheight; i < _height-apoheight; i++){
        int step = i*_width;
        for (int k = 0; k < apowidth; k++){
            mask.emplace_back(std::make_pair(step+k,han_width[k]));
            mask.emplace_back(std::make_pair(step+_width-1-k,han_width[k]));
        }
    }
    return mask;
}

void imageRegistrator::mapCoordinates(std::vector<double> &img, std::vector<double> &output)
{
    // Check that size of map and image are equal
    CHECK_EQ(img.size(),map.size());

    double bgval = percentile<double>(img,1);
    output.resize(img.size());
    for (int i = 0; i<img.size(); ++i)
    {
        double x,y;
        double xr = std::modf(map[i].first,&x);        
        double yr = std::modf(map[i].second,&y);
        double tl = y*_width + x;
        double bl = tl+_width;

        // Check Coordinates are within bounds. Otherwise pad with bgval.
        bool edge[4] = {
            y<_height        &&  y>=0, 
            x<_width         &&  x>=0, 
            (y+1)<_height    &&  (y+1)>=0, 
            (x+1)<_width     &&  (x+1)>=0
        };

        // Map
        double xx =  (edge[0] && edge[1])?img[tl]:bgval;++tl;
        double xy =  (edge[2] && edge[1])?img[bl]:bgval;++bl;
        double xx1 = (edge[0] && edge[3])?img[tl]:bgval;
        double xy1 = (edge[2] && edge[3])?img[bl]:bgval;
       
        output[i] = xx*(1-xr)*(1-yr) + xy*(1-xr)*yr + xx1*xr*(1-yr) +xy1*xr*yr;
    }
}

void imageRegistrator::mapCoordinates(std::vector<double> &img, std::vector<std::complex<double>> &output)
{
    // Check that size of map and image are equal
    CHECK_EQ(img.size(),map.size());

    const double bgval = percentile<double>(img,1);
    // std::vector<double> output(img.size());
    output.resize(img.size());
    #pragma omp parallel for
    for (int i = 0; i<img.size(); ++i)
    {
        double x,y;
        double xr = std::modf(map[i].first,&x);        
        double yr = std::modf(map[i].second,&y);
        double tl = y*_widthd + x;
        double bl = tl+_widthd;

        // Check Coordinates are within bounds. Otherwise pad with bgval.
        bool edge[4] = {
            y<_height        &&  y>=0, 
            x<_width         &&  x>=0, 
            (y+1)<_height    &&  (y+1)>=0, 
            (x+1)<_width     &&  (x+1)>=0
        };

        // Map
        double xx =  (edge[0] && edge[1])?img[tl]:bgval;++tl;
        double xy =  (edge[2] && edge[1])?img[bl]:bgval;++bl;
        double xx1 = (edge[0] && edge[3])?img[tl]:bgval;
        double xy1 = (edge[2] && edge[3])?img[bl]:bgval;
       
        output[i] = xx*(1-xr)*(1-yr) + xy*(1-xr)*yr + xx1*xr*(1-yr) +xy1*xr*yr+0.0i;
    }
}

std::vector<double> imageRegistrator::gaussianHPF (double sigma)
{
    std::vector<double> kernel(_width*_height);
    std::vector<double> kernel_w(_width);
    std::vector<double> kernel_h(_height);
    double sum = getGaussianKernel(kernel_w, _width, sigma);
    double sum2 = getGaussianKernel(kernel_h, _height, sigma);
    // auto max_w = std::max_element(kernel_w.begin(),kernel_w.end());
    // auto max_h = std::max_element(kernel_h.begin(),kernel_h.end());
    double max_val = *std::max_element(kernel_w.begin(),kernel_w.end()) * *std::max_element(kernel_h.begin(),kernel_h.end());

    auto it = kernel.begin();
    for (auto y : kernel_h){
        for (auto x : kernel_w){
            *it++ = 1.0 - x*y/max_val;
        }
    }

    return kernel;
}

template <typename T>
void imageRegistrator::apodize (std::vector<T> &img)
{
    //int aporad = _width *0.12;
    int apowidth = _width * 0.12;
    int apoheight = _height * 0.12;
    std::vector<double> han_width = hanning_window(apowidth*2);
    std::vector<double> han_height = hanning_window(apoheight*2);
    for (int i = 0; i < apoheight; i++) {
        int step = i*_width;
        int rstep = (_height-1-i)*_width;
        // Iterate top and bottom midsection
        for (int k = apowidth; k < _width-apowidth; k++){
            img[step+k] *= han_height[i];
            img[rstep+k] *= han_height[i];
        }
        // Iterate through the 4 corners at one shot (works for even width and height)
        for (int k = 0; k < apowidth; k++) {
            double tmp = han_height[i]*han_width[k];
            img[step+k] *= tmp;
            img[step+_width-1-k] *= tmp;
            img[rstep+k] *= tmp;
            img[rstep+_width-1-k] *= tmp;
        }
    }
    // Iterate left and right midsection
    for (int i = apoheight; i < _height-apoheight; i++){
        int step = i*_width;
        for (int k = 0; k < apowidth; k++){
            img[step+k] *= han_width[k];
            img[step+_width-1-k] *= han_width[k];
        }
    }
}

template void imageRegistrator::apodize<double>(std::vector<double> &img);
template void imageRegistrator::apodize<std::complex<double>>(std::vector<std::complex<double>> &img);

void imageRegistrator::fftShift(std::vector<std::complex<double>> &img, const bool forward )
{
    /* Inspired by https://stackoverflow.com/questions/29226465/fftshift-c-implemetation-for-opencv */ 
    // size of top-left quadrant
    int cx = forward ? (_width + 1) / 2 : _width / 2;
    int cy = forward ? (_height + 1) / 2 : _height / 2;

    for(int i = 0; i < cy ; i++ ) {
        std::swap_ranges(img.begin()+i*_width,img.begin()+i*_width+cx, img.begin()+cx+cy*_width+i*_width);
        std::swap_ranges(img.begin()+i*_width+cx,img.begin()+_width+i*_width,img.begin()+cy*_width+i*_width);
    }
}

void imageRegistrator::phaseCorrelation(std::vector<std::complex<double>> img1,std::vector<std::complex<double>> img2,std::pair<double,double> &results) 
{
    /* Inspired by (keeping links for future reference)
    https://stackoverflow.com/questions/75750139/c-attemping-to-use-stdrotate-with-fftw-complex-data-yields-error-array-mu
    https://stackoverflow.com/questions/9126157/phase-correlation-using-fftw
    */

    std::vector<std::complex<double>> res(_size);
 
    /* Compute FFT of img1 */
    fftw_execute_dft(   fft_forward,
                        reinterpret_cast<fftw_complex*>(img1.data()),
                        reinterpret_cast<fftw_complex*>(img1.data()));

    /* Compute FFT of img2 */
    fftw_execute_dft(   fft_forward,
                        reinterpret_cast<fftw_complex*>(img2.data()),
                        reinterpret_cast<fftw_complex*>(img2.data()));

    /* Compute Cross Power Spectrum */
    #pragma omp parallel for
    for(int i = 0; i < _size ; i++ ) {
        
        res[i] = img2[i]*std::conj(img1[i]);
        res[i] /= std::abs(res[i]);

    }

    /* Compute Phase Correlation */
    fftw_execute_dft(   fft_backward,
                        reinterpret_cast<fftw_complex*>(res.data()),
                        reinterpret_cast<fftw_complex*>(res.data()));

    auto result = std::max_element(res.begin(), res.end(), [](std::complex<double> a, std::complex<double> b)
    {
        return a.real() < b.real();
    });

    centerOfMass(res,std::distance(res.begin(), result),results);

}

template <typename T>
void imageRegistrator::logPolarTransform(std::vector<std::complex<double>> &img, std::vector<T> &output)
{
    // 1.) Apodize image
    //apodize(img);
    for (const auto i : mask){
        img[i.first]*=i.second;
    }
    // 2.) FFT image
    fftw_execute_dft(   fft_forward,
        reinterpret_cast<fftw_complex*>(img.data()),
        reinterpret_cast<fftw_complex*>(img.data()));
	// 3.) FFTShift to HPF
    fftShift(img);
    // 4.) HPF
    std::vector<double> dftHPF(_size);
    #pragma omp parallel for
    for (int i = 0; i < _size; ++i){
        dftHPF[i] = std::abs(img[i]*filter[i]);
    }
    mapCoordinates(dftHPF, output);
}

template void imageRegistrator::logPolarTransform<double>(std::vector<std::complex<double>> &img, std::vector<double> &output);
template void imageRegistrator::logPolarTransform<std::complex<double>>(std::vector<std::complex<double>> &img, std::vector<std::complex<double>> &output);

void imageRegistrator::centerOfMass(const std::vector<std::complex<double>> &img, int m, std::pair<double, double> &com)
{
    // Get Subarray
    const static double col[] = {0,1,2,3,4};
    std::vector<int> xIdx;
    std::vector<int> yIdx;
    int x = m%_width-2;
    int y = m/_width-2;
    for (int j = 0; j<5; ++j,++x,++y){
        xIdx.emplace_back(x<0?x+_width:x);
        yIdx.emplace_back(y<0?y+_height:y);
    }
    double sumX=0;
    double sumY=0;  
    double sum=0;
    for (int j = 0; j<5;++j){
        for (int i = 0; i<5;++i){
            sumX += img[yIdx[j]*_width+xIdx[i]].real()*col[i];
            sumY += img[yIdx[j]*_width+xIdx[i]].real()*col[j];
            sum  += img[yIdx[j]*_width+xIdx[i]].real();
        }
    }
    // minus 5 to compensate for index increments in line -13 (xIdx.emplace_back)
    com.first  = sumY/sum +y-5;
    com.second = sumX/sum +x-5;

}

std::pair<double, double> imageRegistrator::getAngScale(std::vector<std::complex<double>> &img1,std::vector<std::complex<double>> &img2)
{
    std::vector<std::complex<double>> lp1(_size);
    std::vector<std::complex<double>> lp2(_size);
    logPolarTransform(img1,lp1);
    logPolarTransform(img2,lp2);

    std::pair<double,double> results;
    phaseCorrelation(lp1,lp2,results);

    // convert from pixels to angle and scale
    double logbase = std::pow(0.55*_heightd,1.0/_widthd);
    results.first *= 180/_heightd;
    results.second = std::pow(logbase,results.second);

    return results;
}

void imageRegistrator::append(std::vector<std::complex<double>> &img)
{
    queue.push(img);
    logPolarTransform(queue.back(),queue.back());
}

std::pair<double,double> imageRegistrator::getAngScale()
{
    std::pair<double,double> results;
    phaseCorrelation(queue.front(),queue.back(),results);
    // phaseCorrelation(img1,img2,results);

    // convert from pixels to angle and scale
    // double logbase = std::pow(0.55*_heightd,1.0/_widthd);
    results.first *= 180/_heightd;
    results.second = std::pow(logbase,results.second);

    return results;
}