#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <cstddef>
#include <cstdio>

// ============================================================================
// 数据与维度定义
// ============================================================================
constexpr size_t N = 1;
constexpr size_t OC = 64;
constexpr size_t H = 28;
constexpr size_t W = 28;
constexpr size_t IC = 32;
constexpr size_t KH = 3;
constexpr size_t KW = 3;
constexpr size_t IH = H + KH - 1;
constexpr size_t IW = W + KW - 1;

// ============================================================================
// 辅助函数
// ============================================================================
double rtclock() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

/**
 * @brief (最终修正版) 硬编码打印4D张量，以100%匹配MLIR脚本的输出格式。
 */
void print_4d_tensor_for_comparison(const float* data) {
    printf("[[["); // 1. Start with 3 brackets for N, OC, H dimensions
    for (size_t n_ = 0; n_ < N; ++n_) {
        // This loop runs only once
        for (size_t oc_ = 0; oc_ < OC; ++oc_) {
            printf("["); // 2. Bracket for H dimension
            for (size_t h_ = 0; h_ < H; ++h_) {
                printf("["); // 3. Bracket for W dimension
                for (size_t w_ = 0; w_ < W; ++w_) {
                    size_t idx = n_ * (OC * H * W) + oc_ * (H * W) + h_ * W + w_;
                    printf("%g", data[idx]);
                    if (w_ < W - 1) {
                        printf(",     ");
                    }
                }
                printf("]"); // 4. Closing W bracket
                if (h_ < H - 1) {
                    printf(", \n   "); // 5. Separator for H rows
                }
            }
            printf("]"); // 6. Closing H bracket
            if (oc_ < OC - 1) {
                printf(", \n  "); // 7. Separator for OC planes
            }
        }
    }
    printf("]]]\n"); // 8. Closing N and OC brackets
}


// ============================================================================
// 内核函数 (使用正确的计算逻辑)
// ============================================================================
void kernel_conv2d(const float* input, const float* filter, float* output) {
    for (size_t n_ = 0; n_ < N; ++n_) {
        for (size_t oc_ = 0; oc_ < OC; ++oc_) {
            for (size_t h_ = 0; h_ < H; ++h_) {
                for (size_t w_ = 0; w_ < W; ++w_) {
                    size_t output_idx = n_ * (OC * H * W) + oc_ * (H * W) + h_ * W + w_;
                    for (size_t ic_ = 0; ic_ < IC; ++ic_) {
                        for (size_t kh_ = 0; kh_ < KH; ++kh_) {
                            for (size_t kw_ = 0; kw_ < KW; ++kw_) {
                                size_t input_h = h_ + kh_;
                                size_t input_w = w_ + kw_;
                                size_t input_idx = n_ * (IC * IH * IW) + ic_ * (IH * IW) + input_h * IW + input_w;
                                size_t filter_idx = oc_ * (IC * KH * KW) + ic_ * (KH * KW) + kh_ * KW + kw_;
                                output[output_idx] += input[input_idx] * filter[filter_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    std::vector<float> input(N * IC * IH * IW);
    std::vector<float> filter(OC * IC * KH * KW);
    std::vector<float> output(N * OC * H * W);

    std::fill(input.begin(), input.end(), 1.0f);
    std::fill(filter.begin(), filter.end(), 2.0f);
    std::fill(output.begin(), output.end(), 0.0f);

    double t_start = rtclock();
    kernel_conv2d(input.data(), filter.data(), output.data());
    double t_end = rtclock();

    // 使用硬编码的、保证格式正确的打印函数
    print_4d_tensor_for_comparison(output.data());

    double time_s = t_end - t_start;
    printf("%f\n", time_s);

    return 0;
}