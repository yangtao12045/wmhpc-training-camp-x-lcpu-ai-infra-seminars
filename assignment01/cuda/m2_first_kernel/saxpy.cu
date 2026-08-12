#include<cstdio>
#include<cstdlib>
#include<cuda_runtime.h>

#define CUDA_CHECK(call)\
do{\
    cudaError_t err=(call);\
    if(err!=cudaSuccess){\
        fprintf(stderr,"CUDA error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
    }\
}while(0)

__global__ void saxpy(const float* x,float* y,int n){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    int stride=blockDim.x*gridDim.x;
    while(idx<n){
        y[idx]+=x[idx]*2.0f;
        idx+=stride;
    }
}

int main(int argc,char** argv){
    int n=atoi(argv[1]);
    if(n==0){
        printf("SUM=%.0f\n", 0.0);
        return 0;
    }
    size_t bytes=(size_t)n*sizeof(float);
    float *h_x=(float*)malloc(bytes);
    float *h_y=(float*)malloc(bytes);
    if(h_x==nullptr||h_y==nullptr){
        fprintf(stderr,"Host malloc failed\n");
        free(h_x);
        free(h_y);
        return 1;
    }
    float *d_x,*d_y;
    for(int i=0;i<n;++i){
        h_x[i]=((i%2048)-1024)*0.5f;
        h_y[i]=(i%1024)-512;
    }
    CUDA_CHECK(cudaMalloc(&d_x,bytes));
    CUDA_CHECK(cudaMalloc(&d_y,bytes));
    CUDA_CHECK(cudaMemcpy(d_x,h_x,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,h_y,bytes,cudaMemcpyHostToDevice));
    int blocks=(n+255)/256;
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    saxpy<<<blocks,256>>>(d_x,d_y,n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms,start,stop));
    CUDA_CHECK(cudaMemcpy(h_y,d_y,bytes,cudaMemcpyDeviceToHost));
    double sum=0.0;
    for(int i=0;i<n;++i){
        sum+=h_y[i];
    }
    printf("SUM=%.0f\n", sum);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_y);
    free(h_x);
}