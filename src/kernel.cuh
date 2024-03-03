#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include "math.h"

namespace kernel{
    #define pos(x,y,wid) ((y)*(wid)+(x))
    #define max(x,y) ((x)>(y)?(x):(y))
    #define min(x,y) ((x)<(y)?(x):(y))
    #define abs(x) ((x)>0?(x):(-(x)))
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
            if (ipt[pos]<0.) opt[pos*3]=opt[pos*3+1]=opt[pos*3+2] = 0;
            else opt[pos*3]=opt[pos*3+1]=opt[pos*3+2] = min(255,(int)ipt[pos]);
        }
    }
    __global__ void find_variance(double *opt, const double *ipt, const unsigned int knl_size, const unsigned int wid, const unsigned int hei)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y, i,j;

        int anchor=knl_size/2, lx=max(0,x-anchor),rx=min(wid,x+anchor+1),ly=max(0,y-anchor),ry=min(hei,y+anchor+1);

        double all_points=(double)((rx-lx)*(ry-ly)),mse=0.,tempv=0.,avg=0.;

        for (i=lx;i<rx;i++) for (j=ly;j<ry;j++) avg+=ipt[pos(i,j,wid)];

        avg/=all_points;

        for (i=lx;i<rx;i++) for (j=ly;j<ry;j++) tempv=avg-ipt[pos(i,j,wid)],mse+=tempv*tempv;

        mse/=all_points;

        mse=sqrt(mse);
        
        __syncthreads();

        if (x<wid && y<hei) opt[pos(x,y,wid)]=mse;

    }

    __global__ void adaptive_threshold(double *opt, const double *ipt, const unsigned int knl_size, const unsigned int wid, const unsigned int hei)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y, i,j;

        if (x<wid && y<hei){
            int anchor=knl_size/2, lx=max(0,x-anchor),rx=min(wid,x+anchor+1),ly=max(0,y-anchor),ry=min(hei,y+anchor+1);

            double avg=0.;

            for (i=lx;i<rx;i++) for (j=ly;j<ry;j++) avg+=ipt[pos(i,j,wid)];

            avg/=(double)((rx-lx)*(ry-ly));

            double temp=(ipt[pos(x,y,wid)]<avg-1) ?0:ipt[pos(x,y,wid)];

            __syncthreads();

            opt[pos(x,y,wid)]=temp;
        }
    }

    __global__ void multiply(double *opt, const double *ipt, const double *kernel, const unsigned int knl_size, const unsigned int wid, const unsigned int hei)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y, i,j;

        if (x<wid && y<hei){
            int anchor=knl_size/2, lx=max(0,x-anchor),rx=min(wid,x+anchor+1),ly=max(0,y-anchor),ry=min(hei,y+anchor+1);
            double res=0;

            for (i=lx;i<rx;i++) for (j=ly;j<ry;j++) res+=ipt[pos(i,j,wid)]*kernel[pos(i-x+anchor,j-y+anchor,knl_size)];

            opt[pos(x,y,wid)]=res;
        }
    }

    __global__ void square_mean(double *opt, const double *ipt1, const double *ipt2, const unsigned int wid, const unsigned int hei)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x<wid && y<hei){/*
            int p=pos(x,y,wid);
            opt[p]=sqrt(ipt1[p]*ipt1[p]+ipt2[p]*ipt2[p]);*/
        }
    }
}
