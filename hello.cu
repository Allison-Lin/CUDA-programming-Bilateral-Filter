#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}

int main() {
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
