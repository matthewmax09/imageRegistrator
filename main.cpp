#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fftw3.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <complex.h>

/*
void getLogPolar(image)
{   
    // 1. Apodize image
    // 2. DFT image
    // 3. High-pass filter image
    // 4. Perform polar transform
}
*/
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

// https://gist.github.com/lorenzoriano/5414671:
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

void getPolarMap(std::vector<std::pair<double,double>> &map, const double &width, const double &height)
{
    // Ensure map is cleared before adding to it. 
    map.clear();
    double logbase = std::pow(0.55*height,1/width);
    double cx = width/2.0;
    double cy = height/2.0;
    double ellipse_coef = height / width;
    std::vector<double> log(width);
    std::vector<double> logx(width);
    std::vector<double> ang = linspace<double>(0,-M_PI,height,false);
    for (int i = 0; i<log.size(); i++)
    {
        log[i] = std::pow(logbase,i);
        logx[i] = log[i]/ellipse_coef;
    }
    for (auto a : ang){
        for (int i = 0; i<log.size(); i++){
            // x = l*cos(a)+cx
            // y = l*sin(a)+cy
            map.emplace_back(std::make_pair(logx[i]*cos(a)+cx,log[i]*sin(a)+cy));
        }
    }
}

std::vector<double> mapCoordinates(std::vector<double> &img, std::vector<std::pair<double,double>> &map, const int &width, const int &height )
{
    // Check that size of map and image are equal
    CHECK_EQ(img.size(),map.size());

    double bgval = percentile<double>(img,1);
    std::vector<double> output(img.size());
    for (int i = 0; i<img.size(); ++i)
    {
        double x,y;
        double xr = std::modf(map[i].first,&x);        
        double yr = std::modf(map[i].second,&y);
        double tl = y*width + x;
        double bl = tl+width;

        // Check Coordinates are within bounds. Otherwise pad with bgval.
        bool edge[4] = {
            y<height        &&  y>=0, 
            x<width         &&  x>=0, 
            (y+1)<height    &&  (y+1)>=0, 
            (x+1)<width     &&  (x+1)>=0
        };

        // Map
        double xx =  (edge[0] && edge[1])?img[tl]:bgval;++tl;
        double xy =  (edge[2] && edge[1])?img[bl]:bgval;++bl;
        double xx1 = (edge[0] && edge[3])?img[tl]:bgval;
        double xy1 = (edge[2] && edge[3])?img[bl]:bgval;
       
        output[i] = xx*(1-xr)*(1-yr) + xy*(1-xr)*yr + xx1*xr*(1-yr) +xy1*xr*yr;
    }
    return output;
}

double getGaussianKernel(std::vector<double>& result, int n, double sigma)
{
    // CV_Assert(n > 0);
    // Copied and modified from cv::getGaussianKernelBitExact
    //CV_Assert((n & 1) == 1);  // odd

    if (sigma <= 0)
    {
        if (n == 1)
        {
            result = std::vector<double>(1, 1.0);
            return 1.0;
        }
        else if (n == 3)
        {
            double v3[] = {
                0.25,  // 0.25
                0.5,  // 0.5
                0.25   // 0.25
            };
            result.assign(v3, v3 + 3);
            return 1.0;
        }
        else if (n == 5)
        {
            double v5[] = {
                0.0625,  // 0.0625
                0.25,  // 0.25
                0.375,  // 0.375
                0.25,  // 0.25
                0.0625   // 0.0625
            };
            result.assign(v5, v5 + 5);
            return 1.0;
        }
        else if (n == 7)
        {
            double v7[] = {
                0.03125  ,  // 0.03125
                0.109375 ,  // 0.109375
                0.21875  ,  // 0.21875
                0.28125  ,  // 0.28125
                0.21875  ,  // 0.21875
                0.109375 ,  // 0.109375
                0.03125     // 0.03125
            };
            result.assign(v7, v7 + 7);
            return 1.0;
        }
        else if (n == 9)
        {
            double v9[] = {
                4  / 256 ,  // 4  / 256
                13 / 256 ,  // 13 / 256
                30 / 256 ,  // 30 / 256
                51 / 256 ,  // 51 / 256
                60 / 256 ,  // 60 / 256
                51 / 256 ,  // 51 / 256
                30 / 256 ,  // 30 / 256
                13 / 256 ,  // 13 / 256
                4  / 256    // 4  / 256
            };
            result.assign(v9, v9 + 9);
            return 1.0;
        }
    }

    double sd_0_15 = 0.15;  // 0.15
    double sd_0_35 = 0.35;  // 0.35
    double sd_minus_0_125 = -0.125;  // -0.5*0.25

    double sigmaX = sigma > 0 ? double(sigma) : ((double(n)*sd_0_15)+sd_0_35);// double(((n-1)*0.5 - 1)*0.3 + 0.8)
    double scale2X = sd_minus_0_125/(sigmaX*sigmaX);

    int n2_ = (n - 1) / 2;
    std::vector<double> values(n2_ + 1);
    double sum = 0.0;
    for (int i = 0, x = 1 - n; i < n2_; i++, x+=2)
    {
        // x = i - (n - 1)*0.5
        // t = std::exp(scale2X*x*x)
        double t = exp(double(x*x)*scale2X);
        values[i] = t;
        sum += t;
    }
    sum *= double(2);
    //values[n2_] = soft1.0; // x=0 in exp(softdouble(x*x)*scale2X);
    sum += 1.0;
    if ((n & 1) == 0)
    {
        //values[n2_ + 1] = soft1.0;
        sum += 1.0;
    }

    // normalize: sum(k[i]) = 1
    double mul1 = 1.0/sum;

    result.resize(n);

    double sum2 = 0.0;
    for (int i = 0; i < n2_; i++ )
    {
        double t = values[i] * mul1;
        result[i] = t;
        result[n - 1 - i] = t;
        sum2 += t;
    }
    sum2 *= double(2);
    result[n2_] = /*values[n2_]*/ 1.0 * mul1;
    sum2 += result[n2_];
    if ((n & 1) == 0)
    {
        result[n2_ + 1] = result[n2_];
        sum2 += result[n2_];
    }

    return sum2;
}

std::vector<double> gaussianHPF (const int &width, const int &height, double sigma){

    std::vector<double> kernel(width*height);
    std::vector<double> _width(width);
    std::vector<double> _height(height);
    double sum = getGaussianKernel(_width, width, 1);
    double sum2 = getGaussianKernel(_height, height, 1);
    
    auto it = kernel.begin();
    for (auto y : _height){
        for (auto x : _width){
            *it++ = x*y;
        }
    }

    return kernel;
}

std::vector<double> hanning_window(int window_size) {
    std::vector<double> window(window_size);
    for (int i = 0; i < window_size; i++) {
        
        window[i] = 0.5 * (1 - std::cos(2 * M_PI * i / (window_size - 1)));
    
    }
    return window;
}
// Need to precomupte apodization window to simplify the apodization.
void apodize (std::vector<std::complex<double>> &img, const int &width, const int &height)
{
    //int aporad = width *0.12;
    int apowidth = width * 0.12;
    int apoheight = height * 0.12;
    std::vector<double> han_width = hanning_window(apowidth*2);
    std::vector<double> han_height = hanning_window(apoheight*2);
    for (int i = 0; i < apoheight; i++) {
        int step = i*width;
        int rstep = (height-1-i)*width;
        // Iterate top and bottom midsection
        for (int k = apowidth; k < width-apowidth; k++){
            img[step+k] *= han_height[i];
            img[rstep+k] *= han_height[i];
        }
        // Iterate through the 4 corners at one shot (works for even width and height)
        for (int k = 0; k < apowidth; k++) {
            double tmp = han_height[i]*han_width[k];
            img[step+k] *= tmp;
            img[step+width-1-k] *= tmp;
            img[rstep+k] *= tmp;
            img[rstep+width-1-k] *= tmp;
        }
    }
    // Iterate left and right midsection
    for (int i = apoheight; i < height-apoheight; i++){
        int step = i*width;
        for (int k = 0; k < apowidth; k++){
            img[step+k] *= han_width[k];
            img[step+width-1-k] *= han_width[k];
        }
    }
}

void fftShift(std::vector<std::complex<double>> &img, const int &width, const int &height, const bool forward = true)
{
    /* Inspired by https://stackoverflow.com/questions/29226465/fftshift-c-implemetation-for-opencv */ 
    // size of top-left quadrant
    int cx = forward ? (width + 1) / 2 : width / 2;
    int cy = forward ? (height + 1) / 2 : height / 2;

    for(int i = 0, k = 0 ; i < cy ; i++ ) {
        std::swap_ranges(img.begin()+i*width,img.begin()+i*width+cx, img.begin()+cx+cy*width+i*width);
        std::swap_ranges(img.begin()+i*width+cx,img.begin()+width+i*width,img.begin()+cy*width+i*width);
    }
}

void phaseCorrelation(std::vector<std::complex<double>> &img1,std::vector<std::complex<double>> &img2,cv::Mat& result_im)
{
    /* Inspired by (keeping links for future reference)
    https://stackoverflow.com/questions/75750139/c-attemping-to-use-stdrotate-with-fftw-complex-data-yields-error-array-mu
    https://stackoverflow.com/questions/9126157/phase-correlation-using-fftw
    */
    cv::Size sz = result_im.size();
    int height = sz.height;
    int width = sz.width;
    int fft_size = width*height;
    int i;
    // VLOG(1) << cv::typeToString(view1.type()); //CV_32FC1
    
    //std::vector<std::complex<double>> img1(fft_size);
    //std::vector<std::complex<double>> img2(fft_size);
    std::vector<std::complex<double>> res(fft_size);
    
    //fftw_complex *shifted  = ( fftw_complex* )fftw_malloc( sizeof( fftw_complex ) * width * height );

    fftw_plan fft_img1 = fftw_plan_dft_2d( height ,width, reinterpret_cast<fftw_complex*>(img1.data()), 
                                                          reinterpret_cast<fftw_complex*>(img1.data()), 
                                                          FFTW_FORWARD,  FFTW_ESTIMATE );
    fftw_plan fft_img2 = fftw_plan_dft_2d( height ,width, reinterpret_cast<fftw_complex*>(img2.data()), 
                                                          reinterpret_cast<fftw_complex*>(img2.data()), 
                                                          FFTW_FORWARD,  FFTW_ESTIMATE );
    fftw_plan ifft_res = fftw_plan_dft_2d( height ,width, reinterpret_cast<fftw_complex*>(res.data()),  
                                                          reinterpret_cast<fftw_complex*>(res.data()),  
                                                          FFTW_BACKWARD, FFTW_ESTIMATE );

    /* Compute FFT of img1 */
    fftw_execute( fft_img1 );

    /* Compute FFT of img2 */
    fftw_execute( fft_img2 );

    /* Compute Cross Power Spectrum */
    for( i = 0; i < fft_size ; i++ ) {
        
        res[i] = img2[i]*std::conj(img1[i]);
        res[i] /= std::abs(res[i]);

    }

    /* Compute Phase Correlation */
    fftw_execute(ifft_res);

    float* cv_img = ( float* )result_im.data;
    /* Get argmax of ifft*/
    double max_val = 0;
    int max_loc = 0;
    for( i = 0; i < fft_size ; i++ ) {
         if (res[i].real()>max_val){
             max_loc =i;
             max_val = res[i].real();
         }
        *cv_img++ = (float) res[i].real()/fft_size;
    }

    VLOG(1) << "Detected x = " << max_loc%width;
    VLOG(1) << "Detected y = " << max_loc/width;

    /* deallocate FFTW arrays and plans */
    fftw_destroy_plan( fft_img1 );
    fftw_destroy_plan( fft_img2 );
    fftw_destroy_plan( ifft_res );
}

int main(int argc, char* argv[])
{    
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    cv::Mat img141 = cv::imread("/home/airlab/.ros/cropped/Last141.png",0);
    cv::Mat img5 = cv::imread("/home/airlab/.ros/cropped/First5.png",0);
    //VLOG(1) << "image size = " << img5.size[0]*img5.size[1];
    cv::Mat result(img5.size(), CV_32FC1);
    img5.convertTo(img5, CV_32FC1);
    img141.convertTo(img141, CV_32FC1);
    double tx = 100;
    double ty = 50;
    cv::Mat translation_matrix = (cv::Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);
    cv::Mat img141_translated;
    cv::warpAffine(img141, img141_translated, translation_matrix, img141.size());
    cv::imshow("0", img141);
    //cv::imshow("1", img141_translated);
    cv::waitKey(0);
     
    cv::Size sz = img141.size();
    int height = sz.height;
    int width = sz.width;
    int fft_size = width*height;
    int i, j, k;
    unsigned char* p;
    unsigned char* q;
    std::vector<std::complex<double>> img1(height*width);
    std::vector<std::complex<double>> img2(height*width); 
    for( i = 0, k = 0 ; i < height ; i++ ) {
        p = img141.ptr<unsigned char>(i);
        q = img141_translated.ptr<unsigned char>(i);
        for( j = 0 ; j < width ; j++, k++ ) {
            
            img1[k] = ( double ) p[j] + 0.0j;
            img2[k] = ( double ) q[j] + 0.0j;

        }
    }
    std::vector<std::complex<double>> img2_copy = img2;
    phaseCorrelation(img1,img2, result);
    VLOG(1) << "tx = " << tx;
    VLOG(1) << "ty = " << ty;

 }
