#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fftw3.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <complex.h>
#include <iostream>
#include "imageRegistrator.hpp"
#include "utils.hpp"

#include <fstream>
using namespace std::literals;
/*
void getLogPolar(image)
{   
    // 1. Apodize image
    // 2. DFT image
    // 3. High-pass filter image
    // 4. Perform polar transform
}
*/

void complexRealToDoubleMat(std::vector<std::complex<double>> &in, cv::Mat &out)
{
    CHECK_EQ(in.size(),out.total());
    CHECK_EQ(6,out.type());
    double* cv_img = ( double* )out.data;
    int size = in.size();
    for (auto i:in){
        *cv_img++ = i.real();
    }

}

void translateImage(cv::Mat &in, cv::Mat &out, double tx, double ty)
{
    cv::Mat translation_matrix = (cv::Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);
    cv::warpAffine(in, out, translation_matrix, in.size());
}

void print2DArray(std::vector<double> arr, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << arr[i*width+j] << " ";
        }
        std::cout << std::endl;
    }
}

bool areEqual(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat temp;
    cv::bitwise_xor(a,b,temp);
    return !(cv::countNonZero(temp.reshape(1)));
}

int main(int argc, char* argv[])
{    
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    cv::Mat img141 = cv::imread("../images/2231_crop.png",0);
    cv::Mat img5 = cv::imread("../images/2232_crop.png",0);
    cv::Mat result(img5.size(), CV_32FC1);
    //img5.convertTo(img5, CV_32FC1);
    //img141.convertTo(img141, CV_32FC1);
    cv::Mat img141_translated;
    translateImage(img141, img141_translated, 100, 50);
    // cv::imshow("0", img141);
//    cv::imshow("1", img141_translated);
//    cv::waitKey(0);
     
    cv::Size sz = img141.size();
    int height = sz.height;
    int width = sz.width;
    int fft_size = width*height;
    int i, j, k;
    unsigned char* p;
    unsigned char* q;
    std::vector<std::complex<double>> img1(height*width);
    std::vector<std::complex<double>> img2(height*width); 
    std::vector<double> img_map(height*width);
    for( i = 0, k = 0 ; i < height ; i++ ) {
        p = img141.ptr<unsigned char>(i);
        q = img5.ptr<unsigned char>(i);
        // q = img141_translated.ptr<unsigned char>(i);
        for( j = 0 ; j < width ; j++, k++ ) {
            
            img1[k] = ( double ) p[j] + 0.0i;
            img2[k] = ( double ) q[j] + 0.0i;
            img_map[k] = ( double ) p[j];

        }
    }
    std::vector<std::complex<double>> img2_copy = img1;
    cv::Mat swapped(img5);

    imageRegistrator imreg(height,width);
    cv::Mat im1(img5.size(), CV_64FC1);
    cv::Mat im2(img5.size(), CV_64FC1);
    
    complexRealToDoubleMat(img1,im1);
    cv::normalize(im1,im1,255,0,cv::NORM_MINMAX,CV_8UC1);
    cv::imshow("1", im1);
    // cv::imshow("2", im2);
    cv::waitKey(0);
    auto results = imreg.getAngScale(img1,img2);
    VLOG(1) << "Rotation = " << results.first;
    VLOG(1) << "Scale = " << results.second;

}
