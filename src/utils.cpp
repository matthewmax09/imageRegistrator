#include "utils.hpp"

float getGaussianKernel(std::vector<float>& result, int n, float sigma)
{
    CHECK_GT(n,0);
    CHECK_GT(sigma,0);


    if (sigma <= 0)
    {
        if (n == 1)
        {
            result = std::vector<float>(1, 1.0);
            return 1.0;
        }
        else if (n == 3)
        {
            float v3[] = {
                0.25,  // 0.25
                0.5,  // 0.5
                0.25   // 0.25
            };
            result.assign(v3, v3 + 3);
            return 1.0;
        }
        else if (n == 5)
        {
            float v5[] = {
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
            float v7[] = {
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
            float v9[] = {
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

    float sd_0_15 = 0.15;  // 0.15
    float sd_0_35 = 0.35;  // 0.35
    float sd_minus_0_125 = -0.125;  // -0.5*0.25

    float sigmaX = sigma > 0 ? float(sigma) : ((float(n)*sd_0_15)+sd_0_35);// float(((n-1)*0.5 - 1)*0.3 + 0.8)
    float scale2X = sd_minus_0_125/(sigmaX*sigmaX);

    int n2_ = (n - 1) / 2;
    std::vector<float> values(n2_ + 1);
    float sum = 0.0;
    for (int i = 0, x = 1 - n; i < n2_; i++, x+=2)
    {
        // x = i - (n - 1)*0.5
        // t = std::exp(scale2X*x*x)
        float t = exp(float(x*x)*scale2X);
        values[i] = t;
        sum += t;
    }
    sum *= float(2);
    //values[n2_] = soft1.0; // x=0 in exp(softfloat(x*x)*scale2X);
    sum += 1.0;
    if ((n & 1) == 0)
    {
        //values[n2_ + 1] = soft1.0;
        sum += 1.0;
    }

    // normalize: sum(k[i]) = 1
    float mul1 = 1.0/sum;

    result.resize(n);

    float sum2 = 0.0;
    for (int i = 0; i < n2_; i++ )
    {
        float t = values[i] * mul1;
        result[i] = t;
        result[n - 1 - i] = t;
        sum2 += t;
    }
    sum2 *= float(2);
    result[n2_] = /*values[n2_]*/ 1.0 * mul1;
    sum2 += result[n2_];
    if ((n & 1) == 0)
    {
        result[n2_ + 1] = result[n2_];
        sum2 += result[n2_];
    }

    return sum2;
}

std::vector<float> hanning_window(int window_size) 
{
    std::vector<float> window(window_size);
    for (int i = 0; i < window_size; i++) {
        
        window[i] = 0.5 * (1 - std::cos(2 * M_PI * i / (window_size - 1)));
    
    }
    return window;
}

/* 
	https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue
*/

