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
                                                          reinterpret_cast<fftw_complex*>(img1.data()), FFTW_FORWARD,  FFTW_ESTIMATE );
    fftw_plan fft_img2 = fftw_plan_dft_2d( height ,width, reinterpret_cast<fftw_complex*>(img2.data()), 
                                                          reinterpret_cast<fftw_complex*>(img2.data()), FFTW_FORWARD,  FFTW_ESTIMATE );
    fftw_plan ifft_res = fftw_plan_dft_2d( height ,width, reinterpret_cast<fftw_complex*>(res.data()),  
                                                          reinterpret_cast<fftw_complex*>(res.data()),  FFTW_BACKWARD, FFTW_ESTIMATE );

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
    char* p;
    char* q;
    std::vector<std::complex<double>> img1(height*width);
    std::vector<std::complex<double>> img2(height*width); 
    for( i = 0, k = 0 ; i < height ; i++ ) {
        p = img141.ptr<char>(i);
        q = img141_translated.ptr<char>(i);
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
