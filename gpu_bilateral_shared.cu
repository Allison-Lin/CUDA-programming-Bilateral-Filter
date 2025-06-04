// 說明：使用 CUDA 在 GPU 上以shared memory方式實作雙邊濾波器 (Bilateral Filter)。
//       1. 使用 OpenCV 讀入彩色影像 
//       2. 以 CUDA kernel 在 GPU 上做 shared memory 雙邊濾波
//       3. 將結果 copy 回 host 並使用 OpenCV 寫出結果影像
//
// 編譯範例：
//   nvcc `pkg-config --cflags --libs opencv4` gpu_bilateral_shared.cu -o gpu_bilateral_shared
//
// 執行範例：
//   ./gpu_bilateral_shared input.png output.png 5 12.0 16.0
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


// 以 dynamic shared memory 方式，size = (BLOCK_W + 2*radius) * (BLOCK_H + 2*radius) * sizeof(uchar3)
// 將該 tile 載入 shared memory 後，再做雙邊濾波計算，避免重複 global load。
__global__ void bilateralShared(
    const uchar3* __restrict__ src,
    uchar3* __restrict__ dst,
    int width, int height,
    int radius, float sigma_s, float sigma_r)
{
    // 定義 block 大小
    const int BLOCK_W = blockDim.x;
    const int BLOCK_H = blockDim.y;

    // 計算 shared memory tile 參數
    int tileW = BLOCK_W + 2 * radius;   // 含左右各 radius
    int tileH = BLOCK_H + 2 * radius;   // 含上下各 radius
    int sharedSize = tileW * tileH;     // 共有 tileW * tileH 個 uchar3
    
    // 取得 dynamic shared memory 起始指標
    extern __shared__ uchar3 sTile[];   // 一維表示： index = sm_y * tileW + sm_x

    // block 與 thread 的 global 座標
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 先把 tile 載入 shared memory
    // 每個 thread 根據其線性 id i 來讀取／存放 shared memory
    int numThreads = BLOCK_W * BLOCK_H; 
    int threadId  = ty * BLOCK_W + tx;
    
    // 共享記憶體索引 (sm_y, sm_x) 範圍：[0, tileH) × [0, tileW)
    for (int i = threadId; i < sharedSize; i += numThreads) {
        int sm_y = i / tileW;
        int sm_x = i % tileW;
        // 轉換回 global 座標 (gx, gy)：
        int gx = bx * BLOCK_W + (sm_x - radius);
        int gy = by * BLOCK_H + (sm_y - radius);
        // 邊界 clamp
        gx = min(max(gx, 0), width  - 1);
        gy = min(max(gy, 0), height - 1);
        int gidx = gy * width + gx;
        sTile[i] = src[gidx];  // 將 global memory 資料載入到 shared memory
    }

    // 同步：確保所有 tile 元素已載入 shared memory
    __syncthreads();

    // 每個 thread 對應到要處理的像素 (x, y)
    int x = bx * BLOCK_W + tx;  // global x
    int y = by * BLOCK_H + ty;  // global y
    if (x >= width || y >= height) {
        return;  // 若超出影像範圍則跳過
    }
    int outIdx = y * width + x;

    // 取得 shared memory 上的中心像素位置
    int sm_center_x = tx + radius;
    int sm_center_y = ty + radius;
    int sm_center_idx = sm_center_y * tileW + sm_center_x;
    uchar3 center = sTile[sm_center_idx];

    // Pre-calc Gaussian 常數
    float inv_2sigma_s2 = 0.5f / (sigma_s * sigma_s);
    float inv_2sigma_r2 = 0.5f / (sigma_r * sigma_r);

    // 累加變數
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float  norm = 0.0f;

    // 對 radius 範圍內的所有鄰域
    for (int dy = -radius; dy <= radius; ++dy) {
        int sm_y = sm_center_y + dy;  // shared memory y
        for (int dx = -radius; dx <= radius; ++dx) {
            int sm_x = sm_center_x + dx;  // shared memory x
            uchar3 nbr = sTile[sm_y * tileW + sm_x];

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

    // 寫回 global memory
    float inv_norm = 1.0f / norm;
    uchar3 outPixel;
    outPixel.x = static_cast<unsigned char>(sum.x * inv_norm);
    outPixel.y = static_cast<unsigned char>(sum.y * inv_norm);
    outPixel.z = static_cast<unsigned char>(sum.z * inv_norm);
    dst[outIdx] = outPixel;
}


int main(int argc, char* argv[])
{
    if (argc < 6) {
        printf("Usage: %s <input_image> <output_image> <radius> <sigma_s> <sigma_r>\n", argv[0]);
        printf("Example: %s lena.png lena_bi_shared.png 5 12.0 16.0\n", argv[0]);
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
        fprintf(stderr, "Error: 輸入影像必須為 8-bit 3 通道 (BGR)\n");
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
    const int BLOCK_W = 16;
    const int BLOCK_H = 16;
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width  + BLOCK_W - 1) / BLOCK_W,
              (height + BLOCK_H - 1) / BLOCK_H);

    // 計算 dynamic shared memory 大小
    int tileW = BLOCK_W + 2 * radius;
    int tileH = BLOCK_H + 2 * radius;
    size_t sharedMemBytes = static_cast<size_t>(tileW) * static_cast<size_t>(tileH) * sizeof(uchar3);

    // 執行 GPU Kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    bilateralShared<<<grid, block, sharedMemBytes>>>(
        d_src, d_dst, width, height, radius, sigma_s, sigma_r
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
    printf("GPU (shared) elapsed time: %.3f ms\n", gpuTime);

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
