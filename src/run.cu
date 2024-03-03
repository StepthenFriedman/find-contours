#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dbg.cuh"

int main(){
    dbg::find_contours();
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}