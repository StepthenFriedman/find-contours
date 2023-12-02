#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include "lodepng.h"
#include "lodepng.cpp"
//#include "nvcuvid.h"
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>

cudaError_t getVariance(int *opt, const int *ipt, const unsigned int knl_size, const unsigned int ipt_wid, const unsigned int ipt_hei);

__global__ void varKernal(int *opt, const int *ipt, const unsigned int knl_size, const unsigned int ipt_wid, const unsigned int ipt_hei)
{
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y, i, j;
    if (x<(ipt_wid-knl_size+1)&&y<(ipt_hei-knl_size+1)){
        double sum=0.,temp=0.,avg;
        for (i=0;i<knl_size;i++) for (j=0;j<knl_size;j++) sum+=(double)ipt[(y+j)*ipt_wid+(x+i)];
        avg=sum/(double)(knl_size*knl_size);sum=0.;
        for (i=0;i<knl_size;i++) for (j=0;j<knl_size;j++) {
            temp=((double)ipt[(y+j)*ipt_wid+(x+i)]-avg);
            sum+=temp*temp;
        }
        opt[y*(ipt_wid-knl_size+1)+x]=sqrt(sum);
    }
}

__global__ void varKernal3D(int *opt, const int *ipt, const unsigned int knl_size, const unsigned int ipt_wid, const unsigned int ipt_hei)
{
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y, i,j,k;
    const unsigned int opt_wid=ipt_wid-knl_size+1,opt_hei=ipt_hei-knl_size+1;
    double sum=0.,temp=0.,avg=0.;
    if (x<(opt_wid)&&y<(opt_hei)) {
        for (k=0;k<3;k++) {
            avg=0.;
            for (i=0;i<knl_size;i++) for (j=0;j<knl_size;j++) avg+=ipt[((y+i)*ipt_wid+(x+j))*3+k];
            avg/=(double)(knl_size*knl_size);
            for (i=0;i<knl_size;i++) for (j=0;j<knl_size;j++) temp=avg-ipt[((y+i)*ipt_wid+(x+j))*3+k],sum+=temp*temp;
            opt[(y*opt_wid+x)*3+k]=sum;
        }
        sum/=(double)(knl_size*knl_size)*3;
        opt[(y*opt_wid+x)*3]=opt[(y*opt_wid+x)*3+1]=opt[(y*opt_wid+x)*3+2]=sum;
    }
}

__global__ void poolingKernal(int *opt, const int *ipt, const unsigned int knl_size, const unsigned int ipt_wid, const unsigned int ipt_hei)
{
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y, i,j,k;
    const unsigned int opt_wid=ipt_wid-knl_size+1,opt_hei=ipt_hei-knl_size+1;
    double sum=0.,temp=0.,avg=0.;
    if (x<(opt_wid)&&y<(opt_hei)) for (k=0;k<3;k++) {
        avg=0.;
        for (i=0;i<knl_size;i++) for (j=0;j<knl_size;j++) avg+=ipt[((y+i)*ipt_wid+(x+j))*3+k];
        avg/=(double)(knl_size*knl_size);
        opt[(y*opt_wid+x)*3+k]=avg;
    }
}

int dbg2(){
    int i,j,k;
    std::vector<unsigned char> in_image;
    unsigned int ipt_wid,ipt_hei,knl_size=3;
    char error=lodepng::decode(in_image, ipt_wid, ipt_hei, "../resource/Lenna.png",LCT_RGBA);
    const unsigned int opt_wid=ipt_wid-knl_size+1,opt_hei=ipt_hei-knl_size+1;


    printf("size:%lu wid:%u hei:%u\n",in_image.size(),ipt_wid,ipt_hei);
    int* input_image = new int[ipt_wid*ipt_hei*3];
    int* output_image = new int[ipt_wid*ipt_hei*3];
    for (i=0;i<ipt_hei;i++) for (j=0;j<ipt_wid;j++) for (k=0;k<3;k++){
        input_image[(i*ipt_wid+j)*3+k]=in_image[(i*ipt_wid+j)*4+k];
    }
    printf("%s\n",lodepng_error_text(error));

    cudaError_t cudaStatus = getVariance(output_image,input_image,knl_size,ipt_wid,ipt_hei);
    std::vector<unsigned char> out_image;
    
    for (i=0;i<opt_hei;i++) for (j=0;j<opt_wid;j++) {
        for (k=0;k<3;k++){
            out_image.push_back(output_image[(i*opt_wid+j)*3+k]);
        }
        out_image.push_back(255);
    }
    printf("ok!\n");
    error= lodepng::encode("../resource/result.png", out_image, opt_wid, opt_hei);
    printf("%s\n",lodepng_error_text(error));
    return 0;
}
//rgb
int dbg3(){
    int i,j,k;
    const unsigned int opt_wid=512,opt_hei=512;

    std::vector<unsigned char> out_image;
    for (i=0;i<opt_hei;i++) for (j=0;j<opt_wid;j++) {
        if (i<100&&j<100){
            out_image.push_back(100);
            out_image.push_back(0);
            out_image.push_back(0);
            out_image.push_back(255);
        }else{
            out_image.push_back(0);
            out_image.push_back(0);
            out_image.push_back(100);
            out_image.push_back(255);
        }
    }
    printf("ok!\n");
    char error= lodepng::encode("../resource/result.png", out_image, opt_wid, opt_hei);
    printf("%s\n",lodepng_error_text(error));
    return 0;
}

int dbg1(){
    const unsigned int ipt_wid=5,ipt_hei=5,knl_size=3;
    unsigned int i,j;
    const int ipt[ipt_hei][ipt_wid]={
                                    1, 2, 3, 4, 5, 
                                    6, 7, 8, 9, 10, 
                                    11,12,13,14,15, 
                                    16,17,18,19,20,
                                    21,22,23,24,25};
    int opt[ipt_hei-knl_size+1][ipt_wid-knl_size+1]={0};

    cudaError_t cudaStatus = getVariance( &opt[0][0], &ipt[0][0], knl_size, ipt_wid, ipt_hei);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    for (i=0;i<ipt_hei-knl_size+1;i++) {for (j=0;j<ipt_wid-knl_size+1;j++) printf("%d ",opt[i][j]);putchar('\n');}

    return 0;
}

int main(){
    dbg2();
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}

cudaError_t getVariance(int *opt, const int *ipt, const unsigned int knl_size, const unsigned int ipt_wid, const unsigned int ipt_hei)
{
    int *dev_ipt=NULL,*dev_opt=NULL;
    const unsigned int opt_wid=ipt_wid-knl_size+1,opt_hei=ipt_hei-knl_size+1;
    cudaError_t cudaStatus;
    dim3 blocks((opt_wid+31)/32,(opt_hei+31)/32);
    dim3 threadsPerBlock(32,32);
    printf("ipt:(%d,%d)->%d opt:(%d,%d)->%d\n",ipt_wid,ipt_hei,ipt_wid*ipt_hei*3,opt_wid,opt_hei,opt_wid*opt_hei*3);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
 
    cudaStatus = cudaMalloc((void**)&dev_opt, opt_wid*opt_hei*3 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_ipt, ipt_wid*ipt_hei*3 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_ipt, ipt, ipt_wid*ipt_hei*3 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    varKernal3D<<< blocks, threadsPerBlock >>>(dev_opt, dev_ipt, knl_size, ipt_wid, ipt_hei);

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
    cudaFree(dev_opt);
    cudaFree(dev_ipt);
    
    return cudaStatus;
}