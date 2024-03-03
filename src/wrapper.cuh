#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include "kernel.cuh"

namespace wrapper{
    double sobel_x[]={  -1,0,1,
                        -2,0,2,
                        -1,0,1},

           sobel_y[]={  -1,-2,-1,
                         0, 0, 0,
                         1, 2, 1},

           laplacian[]={-1,-1,-1,
                        -1, 8,-1,
                        -1,-1,-1};

    cudaError_t get_grayscale(int *opt, const int *ipt, const unsigned int ipt_wid, const unsigned int ipt_hei)
    {
        int *dev_ipt=NULL,*dev_opt=NULL;

        double* dev_gray=NULL;

        const unsigned int opt_wid=ipt_wid,opt_hei=ipt_hei;

        cudaError_t cudaStatus;
        dim3 blocks_opt((opt_wid+31)/32,(opt_hei+31)/32);
        dim3 threadsPerBlock(32,32);


        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }
    
        cudaStatus = cudaMalloc((void**)&dev_ipt, ipt_wid*ipt_hei*3 * sizeof(int));

        cudaStatus = cudaMalloc((void**)&dev_gray, ipt_wid*ipt_hei * sizeof(double));

        cudaStatus = cudaMalloc((void**)&dev_opt, opt_wid*opt_hei*3 * sizeof(int));

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_ipt, ipt, ipt_wid*ipt_hei*3 * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        kernel::three_channels_to_gray<<< blocks_opt, threadsPerBlock >>>(dev_gray, dev_ipt, ipt_wid, ipt_hei);
        kernel::gray_to_three_channels<<< blocks_opt, threadsPerBlock >>>(dev_opt, dev_gray, ipt_wid, ipt_hei);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "varKernal launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(opt, dev_opt, opt_wid*opt_hei*3 * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

    Error:
        cudaFree(dev_ipt);
        cudaFree(dev_gray);
        cudaFree(dev_opt);

        return cudaStatus;
    }
    #define upper_bounding 20
    #define lower_bounding 4.5
    void dfs(int i,int j,int wid,int hei,double* linker){
        #define x linker[(i-1)*wid+j]
        if (i && x>lower_bounding && x!=255. && x!=0.){
            x=255;
            dfs(i-1,j,wid,hei,linker);
        }
        #undef x
        #define x linker[(i)*wid+j-1]
        if (j && x>lower_bounding && x!=255. && x!=0){
            x=255;
            dfs(i,j-1,wid,hei,linker);
        }
        #undef x
        #define x linker[(i-1)*wid+j-1]
        if (i && j && x>lower_bounding && x!=255. && x!=0){
            x=255;
            dfs(i-1,j-1,wid,hei,linker);
        }
        #undef x
        #define x linker[(i+1)*wid+j]
        if (i<hei-1 && x>lower_bounding && x!=255. && x!=0){
            x=255;
            dfs(i+1,j,wid,hei,linker);
        }
        #undef x
        #define x linker[(i)*wid+j+1]
        if (j<wid-1 && x>lower_bounding && x!=255. && x!=0){
            x=255;
            dfs(i,j+1,wid,hei,linker);
        }
        #undef x
        #define x linker[(i+1)*wid+j+1]
        if (i<hei-1 && j<wid-1 && x>lower_bounding && x!=255. && x!=0){
            x=255;
            dfs(i+1,j+1,wid,hei,linker);
        }
        #undef x
        #define x linker[(i-1)*wid+j+1]
        if (i && j<wid-1 && x>lower_bounding && x!=255. && x!=0){
            x=255;
            dfs(i-1,j+1,wid,hei,linker);
        }
        #undef x
        #define x linker[(i+1)*wid+j-1]
        if (i<hei-1 && j && x>lower_bounding && x!=255. && x!=0){
            x=255;
            dfs(i+1,j-1,wid,hei,linker);
        }
    };
    cudaError_t find_variance(int *opt, const int *ipt, const unsigned int knl_size, const unsigned int wid, const unsigned int hei)
    {

        int *dev_ipt=NULL,*dev_opt=NULL;

        double *dev_gray=NULL,*dev_var=NULL;
        
        cudaError_t cudaStatus;
        dim3 blocks_opt((wid+31)/32,(hei+31)/32);
        dim3 threadsPerBlock(32,32);


        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }
    
        cudaStatus = cudaMalloc((void**)&dev_ipt, wid*hei*3 * sizeof(int));

        cudaStatus = cudaMalloc((void**)&dev_gray, wid*hei * sizeof(double));

        cudaStatus = cudaMalloc((void**)&dev_var, wid*hei * sizeof(double));

        cudaStatus = cudaMalloc((void**)&dev_opt, wid*hei*3 * sizeof(int));

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_ipt, ipt, wid*hei*3 * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        kernel::three_channels_to_gray<<< blocks_opt, threadsPerBlock >>>(dev_gray, dev_ipt, wid, hei);

        kernel::find_variance<<< blocks_opt, threadsPerBlock>>>(dev_var, dev_gray, knl_size, wid, hei);

        kernel::gray_to_three_channels<<< blocks_opt, threadsPerBlock >>>(dev_opt, dev_var, wid, hei);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "varKernal launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(opt, dev_opt, wid*hei*3 * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

    Error:
        cudaFree(dev_ipt);
        cudaFree(dev_gray);
        cudaFree(dev_var);
        cudaFree(dev_opt);

        
        return cudaStatus;
    }
    cudaError_t find_contours(int *opt, const int *ipt, const unsigned int knl_size, const unsigned int wid, const unsigned int hei)
    {

        int *dev_ipt=NULL,*dev_opt=NULL;

        double *dev_gray=NULL,*dev_var=NULL,*linker=(double*)malloc(wid*hei*sizeof(double));
        
        cudaError_t cudaStatus;
        dim3 blocks_opt((wid+31)/32,(hei+31)/32);
        dim3 threadsPerBlock(32,32);


        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }
    
        cudaStatus = cudaMalloc((void**)&dev_ipt, wid*hei*3 * sizeof(int));

        cudaStatus = cudaMalloc((void**)&dev_gray, wid*hei * sizeof(double));

        cudaStatus = cudaMalloc((void**)&dev_var, wid*hei * sizeof(double));

        cudaStatus = cudaMalloc((void**)&dev_opt, wid*hei*3 * sizeof(int));

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_ipt, ipt, wid*hei*3 * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        kernel::three_channels_to_gray<<< blocks_opt, threadsPerBlock >>>(dev_gray, dev_ipt, wid, hei);

        kernel::find_variance<<< blocks_opt, threadsPerBlock>>>(dev_var, dev_gray, knl_size, wid, hei);

        kernel::adaptive_threshold<<<blocks_opt, threadsPerBlock>>>(dev_gray,dev_var,knl_size,wid,hei);

        cudaStatus = cudaMemcpy(linker, dev_gray, wid*hei * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
        #define link(i,j) linker[(i)*wid+(j)]


        for (int i=0;i<hei;i++) for (int j=0;j<wid;j++) if (link(i,j)!=255. && link(i,j)!=0.){
            if (link(i,j)>upper_bounding){
                link(i,j)=255.;
                dfs(i,j,wid,hei,linker);
            }
            else if (x<lower_bounding){
                link(i,j)=0;
            }
        }
        cudaStatus = cudaMemcpy(dev_var,linker,wid*hei * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        kernel::gray_to_three_channels<<< blocks_opt, threadsPerBlock >>>(dev_opt, dev_var, wid, hei);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "varKernal launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(opt, dev_opt, wid*hei*3 * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

    Error:
        cudaFree(dev_ipt);
        cudaFree(dev_gray);
        cudaFree(dev_var);
        cudaFree(dev_opt);

        
        return cudaStatus;
    }
    cudaError_t find_contours_sobel(int *opt, const int *ipt, const unsigned int knl_size, const unsigned int wid, const unsigned int hei)
    {

        int *dev_ipt=NULL,*dev_opt=NULL;

        double *dev_gray=NULL,*dev_x=NULL,*dev_y=NULL,*dev_xy;//*linker=(double*)malloc(wid*hei*sizeof(double));
        
        cudaError_t cudaStatus;
        dim3 blocks_opt((wid+31)/32,(hei+31)/32);
        dim3 threadsPerBlock(32,32);


        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }
    
        cudaStatus = cudaMalloc((void**)&dev_ipt, wid*hei*3 * sizeof(int));

        cudaStatus = cudaMalloc((void**)&dev_gray, wid*hei * sizeof(double));

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_ipt, ipt, wid*hei*3 * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        kernel::three_channels_to_gray<<< blocks_opt, threadsPerBlock >>>(dev_gray, dev_ipt, wid, hei);

        cudaFree(dev_ipt);
        cudaStatus = cudaMalloc((void**)&dev_x, wid*hei * sizeof(double));
        cudaStatus = cudaMalloc((void**)&dev_y, wid*hei * sizeof(double));

        kernel::multiply<<< blocks_opt, threadsPerBlock>>>(dev_x, dev_gray, sobel_x ,knl_size, wid, hei);

        cudaFree(dev_gray);

        kernel::multiply<<< blocks_opt, threadsPerBlock>>>(dev_y, dev_gray, sobel_y ,knl_size, wid, hei);

        cudaStatus = cudaMalloc((void**)&dev_xy, wid*hei * sizeof(double));
        cudaFree()

        kernel::square_mean<<< blocks_opt, threadsPerBlock>>>(dev_xy, dev_x, dev_y ,wid, hei);

        cudaStatus = cudaMalloc((void**)&dev_opt, wid*hei*3 * sizeof(int));
        kernel::gray_to_three_channels<<< blocks_opt, threadsPerBlock >>>(dev_opt, dev_x, wid, hei);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernal launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(opt, dev_opt, wid*hei*3 * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

    Error:
        cudaFree(dev_x);
        cudaFree(dev_y);
        cudaFree(dev_xy);
        cudaFree(dev_opt);

        
        return cudaStatus;
    }
    /*
    cudaError_t test(int *opt, const int *ipt, const unsigned int wid, const unsigned int hei)
    {
        int *dev_ipt=NULL,*dev_opt=NULL;

        double *dev_gray=NULL,*dev_sobel_x=NULL,*dev_sobel_y=NULL;
        
        cudaError_t cudaStatus;
        dim3 blocks_opt((wid+31)/32,(hei+31)/32);
        dim3 threadsPerBlock(32,32);


        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }
    
        cudaStatus = cudaMalloc((void**)&dev_ipt, wid*hei*3 * sizeof(int));

        cudaStatus = cudaMalloc((void**)&dev_gray, wid*hei * sizeof(double));

        cudaStatus = cudaMalloc((void**)&dev_sobel_x, 3*3* sizeof(double));

        cudaStatus = cudaMalloc((void**)&dev_sobel_y, 3*3* sizeof(double));

        cudaStatus = cudaMalloc((void**)&dev_opt, wid*hei*3 * sizeof(int));

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_ipt, ipt, wid*hei*3 * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        kernel::three_channels_to_gray<<< blocks_opt, threadsPerBlock >>>(dev_gray, dev_ipt, wid, hei);



        //kernel::find_variance<<< blocks_opt, threadsPerBlock >>>(dev_var, dev_gray, knl_size,wid, hei);

        //kernel::adaptive_threshold<<< blocks_opt, threadsPerBlock >>>(dev_thresh, dev_var, knl_size, wid, hei);

        kernel::gray_to_three_channels<<< blocks_opt, threadsPerBlock >>>(dev_opt, dev_thresh, wid, hei);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "varKernal launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(opt, dev_opt, wid*hei*3 * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

    Error:
        cudaFree(dev_ipt);
        cudaFree(dev_gray);
        cudaFree(dev_var);
        cudaFree(dev_lr);
        cudaFree(dev_thresh);
        cudaFree(dev_opt);

        
        return cudaStatus;
    }*/
}