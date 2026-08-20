#include<cuda_fp8.h>
#include"../common.h"
#include <random>
#include <cstdlib>

__global__ void mma_kernel(const __nv_fp8_e4m3* A,const __nv_fp8_e4m3* B,float *D){
    int lane=threadIdx.x;
    int group=lane>>2;
    int tig=lane&3;

    __nv_fp8_e4m3 a0=A[group*32+tig*4];
    __nv_fp8_e4m3 a1=A[group*32+tig*4+1];
    __nv_fp8_e4m3 a2=A[group*32+tig*4+2];
    __nv_fp8_e4m3 a3=A[group*32+tig*4+3];
    __nv_fp8_e4m3 a4=A[(group+8)*32+tig*4];
    __nv_fp8_e4m3 a5=A[(group+8)*32+tig*4+1];
    __nv_fp8_e4m3 a6=A[(group+8)*32+tig*4+2];
    __nv_fp8_e4m3 a7=A[(group+8)*32+tig*4+3];
    __nv_fp8_e4m3 a8=A[group*32+tig*4+16];
    __nv_fp8_e4m3 a9=A[group*32+tig*4+17];
    __nv_fp8_e4m3 a10=A[group*32+tig*4+18];
    __nv_fp8_e4m3 a11=A[group*32+tig*4+19];
    __nv_fp8_e4m3 a12=A[(group+8)*32+tig*4+16];
    __nv_fp8_e4m3 a13=A[(group+8)*32+tig*4+17];
    __nv_fp8_e4m3 a14=A[(group+8)*32+tig*4+18];
    __nv_fp8_e4m3 a15=A[(group+8)*32+tig*4+19];
    unsigned ra[4];
    ra[0]=((unsigned)a0.__x)|((unsigned)a1.__x<<8)|((unsigned)a2.__x<<16)|((unsigned)a3.__x<<24);
    ra[1]=((unsigned)a4.__x)|((unsigned)a5.__x<<8)|((unsigned)a6.__x<<16)|((unsigned)a7.__x<<24);
    ra[2]=((unsigned)a8.__x)|((unsigned)a9.__x<<8)|((unsigned)a10.__x<<16)|((unsigned)a11.__x<<24);
    ra[3]=((unsigned)a12.__x)|((unsigned)a13.__x<<8)|((unsigned)a14.__x<<16)|((unsigned)a15.__x<<24);

    __nv_fp8_e4m3 b0=B[(tig*4)*8+group];
    __nv_fp8_e4m3 b1=B[(tig*4+1)*8+group];
    __nv_fp8_e4m3 b2=B[(tig*4+2)*8+group];
    __nv_fp8_e4m3 b3=B[(tig*4+3)*8+group];
    __nv_fp8_e4m3 b4=B[(tig*4+16)*8+group];
    __nv_fp8_e4m3 b5=B[(tig*4+17)*8+group];
    __nv_fp8_e4m3 b6=B[(tig*4+18)*8+group];
    __nv_fp8_e4m3 b7=B[(tig*4+19)*8+group];
    unsigned rb[2];
    rb[0]=((unsigned)b0.__x)|((unsigned)b1.__x<<8)|((unsigned)b2.__x<<16)|((unsigned)b3.__x<<24);
    rb[1]=((unsigned)b4.__x)|((unsigned)b5.__x<<8)|((unsigned)b6.__x<<16)|((unsigned)b7.__x<<24);

    float c[4]={0.f,0.f,0.f,0.f},d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        :"r"(ra[0]),"r"(ra[1]),"r"(ra[2]),"r"(ra[3]),"r"(rb[0]),
        "r"(rb[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));

        D[group*8+tig*2]=d[0];
        D[group*8+tig*2+1]=d[1];
        D[(group+8)*8+tig*2]=d[2];
        D[(group+8)*8+tig*2+1]=d[3];
}

static int run_path(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 15);

    __nv_fp8_e4m3 hA[16 * 32];
    __nv_fp8_e4m3 hB[32 * 8];

    float fA[16 * 32];
    float fB[32 * 8];
    float ref[16 * 8] = {};

    // Generate exactly representable small integers.
    for (int i = 0; i < 16 * 32; ++i) {
        hA[i] = __nv_fp8_e4m3((float)(dist(rng) - 8));
        fA[i] = float(hA[i]);
    }

    for (int i = 0; i < 32 * 8; ++i) {
        hB[i] = __nv_fp8_e4m3((float)(dist(rng) - 8));
        fB[i] = float(hB[i]);
    }

    // CPU reference: D = A(16x32) * B(32x8)
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < 32; ++k) {
                sum += fA[m * 32 + k] * fB[k * 8 + n];
            }
            ref[m * 8 + n] = sum;
        }
    }

    __nv_fp8_e4m3 *dA, *dB;
    float *dD;

    CUDA_CHECK(cudaMalloc((void**)&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc((void**)&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc((void**)&dD, sizeof(ref)));

    CUDA_CHECK(cudaMemcpy(
        dA, hA, sizeof(hA), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        dB, hB, sizeof(hB), cudaMemcpyHostToDevice));

    mma_kernel<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();

    float got[16 * 8];

    CUDA_CHECK(cudaMemcpy(
        got, dD, sizeof(got), cudaMemcpyDeviceToHost));

    int bad = 0;

    for (int i = 0; i < 16 * 8; ++i) {
        if (got[i] != ref[i]) {
            ++bad;
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);

    return bad;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("FAIL expected seed argument\n");
        return 1;
    }

    unsigned seed = (unsigned)std::strtoul(argv[1], nullptr, 10);

    int bad = run_path(seed);

    if (bad == 0) {
        printf("PASS seed=%u\n", seed);
        return 0;
    }

    printf("MISMATCH seed=%u bad=%d\n", seed, bad);
    return 1;
}