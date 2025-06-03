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

/*
void getLogPolar(image)
{   
    // 1. Apodize image
    // 2. DFT image
    // 3. High-pass filter image
    // 4. Perform polar transform
}
*/

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

    cv::Mat img141 = cv::imread("/home/airlab/.ros/cropped/Last141.png",0);
    cv::Mat img5 = cv::imread("/home/airlab/.ros/cropped/First5.png",0);
    //VLOG(1) << "image size = " << img5.size[0]*img5.size[1];
    cv::Mat result(img5.size(), CV_32FC1);
    //img5.convertTo(img5, CV_32FC1);
    //img141.convertTo(img141, CV_32FC1);
    double tx = 100;
    double ty = 50;
    cv::Mat translation_matrix = (cv::Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);
    cv::Mat img141_translated;
    cv::warpAffine(img141, img141_translated, translation_matrix, img141.size());
    //cv::imshow("0", img141);
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
        q = img141_translated.ptr<unsigned char>(i);
        for( j = 0 ; j < width ; j++, k++ ) {
            
            img1[k] = ( double ) p[j] + 0.0j;
            img2[k] = ( double ) q[j] + 0.0j;
            img_map[k] = ( double ) p[j];

        }
    }
    std::vector<std::complex<double>> img2_copy = img1;
    // phaseCorrelation(img1,img2, result);
//    VLOG(1) << "tx = " << tx;
//    VLOG(1) << "ty = " << ty;
    cv::Mat swapped(img5);
    // std::fill(img2_copy.begin(),img2_copy.end(),255+0.0j);
//    fftShift(img2_copy,width,height); 
    // char* cv_img = ( char* )swapped.data;
    // for( i = 0; i < fft_size ; i++ ) {
    //     *cv_img++ = (char) img2_copy[i].real();
    // }
    // cv::imshow("0", swapped);
    // cv::waitKey(0);
//    fftShift(img2_copy,width,height,false);
//    cv_img = ( char* )swapped.data;
//    for( i = 0; i < fft_size ; i++ ) {
//        *cv_img++ = (char) img2_copy[i].real();
//    }
//    cv::imshow("0", swapped);
//    cv::waitKey(0);
    // apodize(img2_copy,width,height);
    // cv_img = ( char* )swapped.data;
    // for( i = 0; i < fft_size ; i++ ) {
    //     *cv_img++ = (char) img2_copy[i].real();
    // }
    //cv::imshow("0", swapped);
    //cv::waitKey(0);

    // for(auto i:aaa){VLOG(1) << i;}
    imageRegistrator imreg(height,width);
    // std::vector<double> img2_polar = imreg.mapCoordinates(img_map);
    std::fill(img2_copy.begin(),img2_copy.end(),255+0.0j);
    // imreg.apodize(img2_copy);
    img_map = imreg.gaussianHPF(22);
    cv::Mat im1(img5.size(), CV_64FC1);
    cv::Mat im2(img5.size(), CV_8UC1);
     double* cv_img = ( double* )im1.data;
     for( i = 0; i < fft_size ; i++ ) {
         *cv_img++ = img_map[i];
     }

    uchar* cv_img1 = ( uchar* )im2.data;
    for( i = 0; i < fft_size ; i++ ) {
        // *cv_img1++ = cv::saturate_cast<uchar>(img_map[i]*255);
        *cv_img1++ = roundCast<uchar>(img_map[i]*255);
    }
    cv::normalize(im1,im1,255,0,cv::NORM_MINMAX,CV_8UC1);
    // bool eq = std::equal(result2.begin<uchar>(), result2.end<uchar>(), swapped.begin<uchar>());
    // VLOG(1)<< eq;
    VLOG(1) << areEqual(im1,im2);

    imreg.phaseCorrelation(img1,img2);
    // cv::imshow("0", result2);
    // cv::waitKey(0);
    // for (auto i:map){VLOG(1)<< i.first;}

    // imageRegistrator imreg(height,width);

    // cv::imwrite("/home/airlab/cpp_img.png",swapped);

    // std::ofstream outfile{"test.bin", std::ios::binary};
    // outfile.write(reinterpret_cast<const char *>(img_map.data()),
    //                 img_map.size() * sizeof(decltype(img_map)::value_type));
    // outfile.close();

    // VLOG(1)<< "Class height: " << imreg.getHeight();
    // VLOG(1)<< "Class width: " << imreg.getWidth();

}
