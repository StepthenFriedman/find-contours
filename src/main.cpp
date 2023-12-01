#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <./variance.cu>

int main()
{
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

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}