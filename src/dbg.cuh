#pragma once
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include "lodepng.h"
#include "lodepng.cpp"
#include "wrapper.cuh"

namespace dbg{
    int get_grayscale(){
        int i,j,k;
        std::vector<unsigned char> in_image;
        unsigned int ipt_wid,ipt_hei;
        char error=lodepng::decode(in_image, ipt_wid, ipt_hei, "../resource/input/Lenna.png",LCT_RGBA);
        const unsigned int opt_wid=ipt_wid,opt_hei=ipt_hei;


        printf("%d:%s\n",error,lodepng_error_text(error));
        printf("size:%lu wid:%u hei:%u\n",in_image.size(),ipt_wid,ipt_hei);
        int* input_image = new int[ipt_wid*ipt_hei*3];
        int* output_image = new int[opt_wid*opt_hei*3];

        for (i=0;i<ipt_hei;i++) for (j=0;j<ipt_wid;j++) for (k=0;k<3;k++){
            input_image[(i*ipt_wid+j)*3+k]=in_image[(i*ipt_wid+j)*4+k];
        }

        cudaError_t cudaStatus = wrapper::get_grayscale(output_image,input_image,ipt_wid,ipt_hei);

        std::vector<unsigned char> out_image;
        
        for (i=0;i<opt_hei;i++) for (j=0;j<opt_wid;j++) {
            for (k=0;k<3;k++){
                out_image.push_back(output_image[(i*opt_wid+j)*3+k]);
            }
            out_image.push_back(255);
        }
        printf("ok!\n");
        error= lodepng::encode("../resource/output/Lenna_gray.png", out_image, opt_wid, opt_hei);
        printf("%s\n",lodepng_error_text(error));
        return 0;
    }
    int find_variance(){
        int i,j,k;
        std::vector<unsigned char> in_image;
        unsigned int ipt_wid,ipt_hei,knl_size=3;
        char error=lodepng::decode(in_image, ipt_wid, ipt_hei, "../resource/input/Lenna_avged.png",LCT_RGBA);
        const unsigned int opt_wid=ipt_wid-knl_size+1,opt_hei=ipt_hei-knl_size+1;


        printf("%d:%s\n",error,lodepng_error_text(error));
        printf("size:%lu wid:%u hei:%u\n",in_image.size(),ipt_wid,ipt_hei);
        int* input_image = new int[ipt_wid*ipt_hei*3];
        int* output_image = new int[opt_wid*opt_hei*3];
        for (i=0;i<ipt_hei;i++) for (j=0;j<ipt_wid;j++) for (k=0;k<3;k++){
            input_image[(i*ipt_wid+j)*3+k]=in_image[(i*ipt_wid+j)*4+k];
        }

        cudaError_t cudaStatus = wrapper::find_variance(output_image,input_image,knl_size,ipt_wid,ipt_hei);
        std::vector<unsigned char> out_image;
        
        for (i=0;i<opt_hei;i++) for (j=0;j<opt_wid;j++) {
            for (k=0;k<3;k++){
                out_image.push_back(output_image[(i*opt_wid+j)*3+k]);
            }
            out_image.push_back(255);
        }
        printf("ok!\n");
        error= lodepng::encode("../resource/output/Lenna-var.png", out_image, opt_wid, opt_hei);
        printf("%s\n",lodepng_error_text(error));
        return 0;
    }
}
