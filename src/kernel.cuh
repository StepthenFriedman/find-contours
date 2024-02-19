#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

namespace kernel{
    #define pos(x,y,wid) ((y)*(wid)+(x))
    #define abssub(x,y) (((x)>(y))?((x)-(y)):((y)-(x)))
    /*
    The horizontal axis is x and the vertical axis is y (0<=x<ipt_wid, 0<=y<ipt_hei).
    Three values adjacent horizontally (on x axis) belong to the same point.
    ipt_wid and ipt_height refers to the width and height of points, not values.
    for values, use (ipt_wid*3) and ipt_height.
    the x used in kernel function is the x-axis coordinate of Points
    */
    __global__ void three_channels_to_gray(double *opt, const int *ipt, const unsigned int ipt_wid, const unsigned int ipt_hei)
    {
        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x<(ipt_wid)&&y<(ipt_hei)){
            unsigned int pos=pos(x,y,ipt_wid);
            opt[pos]=(double)(ipt[pos*3]+ipt[pos*3+1]+ipt[pos*3+2])/3.;
        }
    }

    __global__ void gray_to_three_channels(int *opt, const double *ipt, const unsigned int ipt_wid, const unsigned int ipt_hei)
    {
        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x<(ipt_wid)&&y<(ipt_hei)){
            unsigned int pos=pos(x,y,ipt_wid);
            opt[pos*3]=opt[pos*3+1]=opt[pos*3+2] = ipt[pos];
        }
    }

    __global__ void find_variance(double *opt, const double *ipt, const unsigned int knl_size, const unsigned int ipt_wid, const unsigned int ipt_hei)
    {
        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y, i,j;
        const unsigned int opt_wid=ipt_wid-knl_size+1,opt_hei=ipt_hei-knl_size+1;
        if (x<(opt_wid)&&y<(opt_hei)) {

            double mse=0.,temp=0.,avg=0.;

            for (i=0;i<knl_size;i++) for (j=0;j<knl_size;j++) avg+=ipt[pos(x+i,y+j,ipt_wid)];

            avg/=(double)(knl_size*knl_size);

            for (i=0;i<knl_size;i++) for (j=0;j<knl_size;j++) temp=avg-ipt[pos(x+i,y+j,ipt_wid)],mse+=temp*temp;

            mse/=(double)(knl_size*knl_size);

            mse=sqrt(mse);

            opt[pos(x,y,opt_wid)]=mse;
        }
    }

    __global__ void get_linear_regression(double *opt, const double *ipt, const unsigned int knl_size, const unsigned int ipt_wid, const unsigned int ipt_hei)
    {
        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y, i,j;
        const unsigned int opt_wid=ipt_wid-knl_size+1,opt_hei=ipt_hei-knl_size+1,anchor=kernel_size/2;
        if (x<(opt_wid)&&y<(opt_hei)) {

            double sum_x=0.,sum_y=0.;

            for (i=0;i<knl_size;i++) for (j=0;j<knl_size;j++) {
                sum_x+=(x+i-anchor)*ipt[pos(x+i,y+j,ipt_wid)];
                sum_y+=(y+j-anchor)*ipt[pos(x+i,y+j,ipt_wid)];
            }

            double k=sum_y/sum_x,base=sqrt(1.+(k*k)),mse;

            for (i=0;i<knl_size;i++) for (j=0;j<knl_size;j++) {

            }

            opt[pos(x,y,opt_wid)]=sum;
        }
    }
}
