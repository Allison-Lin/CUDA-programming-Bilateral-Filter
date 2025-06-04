// 說明：使用 CUDA 在 GPU 上以global memory方式實作雙邊濾波器（Bilateral Filter）。
//       1. 使用 OpenCV 讀入彩色影像 
//       2. 以 CUDA kernel 在 GPU 上做 global memory 雙邊濾波
//       3. 將結果 copy 回 host 並使用 OpenCV 寫出結果影像
//
// 編譯範例：
//   nvcc `pkg-config --cflags --libs opencv4` gpu_bilateral_global.cu -o gpu_bilateral_global
//
// 執行範例：
//   ./gpu_bilateral_global input.png output.png 5 12.0 16.0
//   參數說明：
//     input.png   → 輸入影像路徑
//     output.png  → 輸出影像路徑
//     5           → filter 半徑 radius (e.g. 5 => kernel 大小 11×11)
//     12.0        → sigma_s (空間域標準差)
//     16.0        → sigma_r (色彩域標準差)

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define CUDA_CHECK(err)                                            \
    do {                                                           \
        cudaError_t err__ = (err);                                 \
        if (err__ != cudaSuccess) {                                \
            fprintf(stderr, "[CUDA ERROR] %s (line %d): %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err__));\
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// global-memory bilateral kernel
__global__ void bilateralGlobal(
    const uchar3* __restrict__ src,
    uchar3* __restrict__ dst,
    int width, int height,
    int radius, float sigma_s, float sigma_r)
{
    // 計算當前 thread 對應到影像座標 (x, y)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uchar3 center = src[idx];

    // 參數預先計算
    float inv_2sigma_s2 = 0.5f / (sigma_s * sigma_s);
    float inv_2sigma_r2 = 0.5f / (sigma_r * sigma_r);

    // 累加變數
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float  norm = 0.0f;

    // 以 global memory 方式，對半徑範圍內所有鄰域做操作
    for (int dy = -radius; dy <= radius; ++dy) {
        int yy = y + dy;
        // 邊界處理：clamp 到合法範圍
        yy = min(max(yy, 0), height - 1);

        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = x + dx;
            xx = min(max(xx, 0), width - 1);

            int nidx = yy * width + xx;
            uchar3 nbr = src[nidx];

            // 計算Gaussian空間距離權重
            float dsq = float(dx * dx + dy * dy);
            float w_s = expf(-dsq * inv_2sigma_s2);

            // 計算Gaussian範圍像素色差權重 
            float dr = float(float(nbr.x) - float(center.x));
            float dg = float(float(nbr.y) - float(center.y));
            float db = float(float(nbr.z) - float(center.z));
            float rsq = dr * dr + dg * dg + db * db;
            float w_r = expf(-rsq * inv_2sigma_r2);

            float w = w_s * w_r;

            sum.x += w * float(nbr.x);
            sum.y += w * float(nbr.y);
            sum.z += w * float(nbr.z);
            norm  += w;
        }
    }

    // 寫回結果 
    float inv_norm = 1.0f / norm;
    uchar3 outPixel;
    outPixel.x = static_cast<unsigned char>(sum.x * inv_norm);
    outPixel.y = static_cast<unsigned char>(sum.y * inv_norm);
    outPixel.z = static_cast<unsigned char>(sum.z * inv_norm);

    dst[idx] = outPixel;
}

int main(int argc, char* argv[])
{
    if (argc < 6) {
        printf("Usage: %s <input_image> <output_image> <radius> <sigma_s> <sigma_r>\n", argv[0]);
        printf("Example: %s lena.png lena_bi.png 5 12.0 16.0\n", argv[0]);
        return -1;
    }

    const char*  input_path  = argv[1];
    const char*  output_path = argv[2];
    int          radius      = atoi(argv[3]);
    float        sigma_s     = atof(argv[4]);
    float        sigma_r     = atof(argv[5]);

    cv::Mat img_bgr = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
        fprintf(stderr, "Error: 無法讀取影像 %s\n", input_path);
        return -1;
    }
    int width  = img_bgr.cols;
    int height = img_bgr.rows;

    if (img_bgr.type() != CV_8UC3) {
        fprintf(stderr, "Error: 輸入影像必須為 8-bit 3 channels (BGR)\n");
        return -1;
    }

    // 在 Host 端建立輸出 Mat，並把資料轉成 uchar3 指標
    cv::Mat out_bgr(height, width, CV_8UC3);
    uchar3* h_src = reinterpret_cast<uchar3*>(img_bgr.data);
    uchar3* h_dst = reinterpret_cast<uchar3*>(out_bgr.data);

    size_t numPixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    size_t bufBytes  = numPixels * sizeof(uchar3);

    // 在 Device 上分配記憶體
    uchar3* d_src = nullptr;
    uchar3* d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bufBytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bufBytes));

    // Host → Device copy
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bufBytes, cudaMemcpyHostToDevice));

    // 設定 CUDA 執行緒佈局
    dim3 block(32, 32);       //每個 block 有 1024 個 threads 形成 32x32 的二維排列
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // 執行 GPU Kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    bilateralGlobal<<<grid, block>>>(d_src, d_dst, width, height, radius, sigma_s, sigma_r);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
    printf("GPU (global) elapsed time: %.3f ms\n", gpuTime);

    // Device → Host copy結果
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, bufBytes, cudaMemcpyDeviceToHost));

    bool success = cv::imwrite(output_path, out_bgr);
    if (!success) {
        fprintf(stderr, "Error: 無法寫出影像 %s\n", output_path);
    } else {
        printf("Result saved to %s\n", output_path);
    }

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
