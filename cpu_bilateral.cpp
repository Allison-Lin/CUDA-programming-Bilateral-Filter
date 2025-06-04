// 說明：使用 OpenCV 在 CPU 上以單執行緒做雙邊濾波器 (Bilateral Filter) 
//
// 編譯範例：
//   g++ cpu_bilateral.cpp -o cpu_bilateral `pkg-config --cflags --libs opencv4` -std=c++17
//
// 執行範例：
//   ./cpu_bilateral input.png output_cpu.png 5 12.0 16.0
//   參數說明：
//     input.png   → 輸入影像路徑
//     output_cpu.png → 輸出結果影像路徑
//     5           → filter 半徑 radius (e.g. 5 → diameter = 11)
//     12.0        → sigma_s (空間域標準差)
//     16.0        → sigma_r (色彩域標準差)

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    if (argc < 6) {
        printf("Usage: %s <input_image> <output_image> <radius> <sigma_s> <sigma_r>\n", argv[0]);
        return -1;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];
    int    radius  = std::atoi(argv[3]);
    double sigma_s = std::atof(argv[4]);
    double sigma_r = std::atof(argv[5]);

    // OpenCV bilateralFilter 使用的 kernel 直徑
    int diameter = 2 * radius + 1;  

    cv::Mat img_bgr = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
        fprintf(stderr, "Error: 無法讀取影像 %s\n", input_path);
        return -1;
    }
    if (img_bgr.type() != CV_8UC3) {
        fprintf(stderr, "Error: 輸入影像必須為 8-bit 3 通道格式 (BGR)\n");
        return -1;
    }

    cv::Mat out_bgr(img_bgr.rows, img_bgr.cols, CV_8UC3);

    // 強制 OpenCV 使用單執行緒
    cv::setNumThreads(1);

    // 執行 CPU 上的雙邊濾波並計時
    auto t0 = std::chrono::high_resolution_clock::now();

    // cv::bilateralFilter(src, dst, diameter, sigmaColor, sigmaSpace);
    cv::bilateralFilter(
        img_bgr,         // 輸入影像
        out_bgr,         // 輸出影像
        diameter,        // kernel 直徑 (2*radius+1)
        sigma_r,         // sigmaColor (色彩域標準差)
        sigma_s,         // sigmaSpace (空間域標準差)
        cv::BORDER_REPLICATE
    );

    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU (single-thread) elapsed time: %.3f ms\n", cpuTime);

    // 寫出結果影像
    bool success = cv::imwrite(output_path, out_bgr);
    if (!success) {
        fprintf(stderr, "Error: 無法寫出影像 %s\n", output_path);
        return -1;
    }
    printf("Result saved to %s\n", output_path);

    return 0;
}
