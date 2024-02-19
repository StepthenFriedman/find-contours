#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include "kernel.cuh"

namespace wrapper{

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

    cudaError_t find_variance(int *opt, const int *ipt, const unsigned int knl_size, const unsigned int ipt_wid, const unsigned int ipt_hei)
    {
        int *dev_ipt=NULL,*dev_opt=NULL;

        double *dev_gray=NULL,*dev_var=NULL;

        const unsigned int opt_wid=ipt_wid-knl_size+1,opt_hei=ipt_hei-knl_size+1;
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

        cudaStatus = cudaMalloc((void**)&dev_var, ipt_wid*ipt_hei * sizeof(double));

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

        kernel::find_variance<<< blocks_opt, threadsPerBlock >>>(dev_var, dev_gray, knl_size,ipt_wid, ipt_hei);

        kernel::gray_to_three_channels<<< blocks_opt, threadsPerBlock >>>(dev_opt, dev_var, ipt_wid, ipt_hei);

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
        cudaFree(dev_var);
        cudaFree(dev_opt);

        
        return cudaStatus;
    }
}