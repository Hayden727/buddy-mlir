#include <iostream>
#include <vector>
#include <chrono>
#include <numeric> // For std::fill
#include <cstddef> // For size_t
#include <cstdio>  // For printf
#include <cmath>   // For std::exp
#include <limits>  // For std::numeric_limits

// ============================================================================
// 数据与维度定义 (与 MLIR 版本保持一致)
// ============================================================================
constexpr size_t BATCH_SIZE = 16;
constexpr size_t FEATURES_SIZE = 1024;

// ============================================================================
// 辅助函数 (与之前版本相同)
// ============================================================================

double rtclock() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

/**
 * @brief 打印二维矩阵，以匹配MLIR测试脚本的输出格式。
 *        格式: [[val00, val01, ...], [val10, val11, ...], ...]
 * @param data 指向矩阵数据的指针。
 * @param rows 矩阵的行数。
 * @param cols 矩阵的列数。
 */
void print_matrix_for_comparison(const float* data, size_t rows, size_t cols) {
    printf("[[");
    for (size_t r = 0; r < rows; ++r) {
        if (r > 0) {
            printf("], \n [");
        } else {
            printf("[");
        }
        for (size_t c = 0; c < cols; ++c) {
            printf("%g", data[r * cols + c]);
            if (c < cols - 1) {
                printf(", ");
            }
        }
    }
    printf("]]\n");
}


// ============================================================================
// 内核函数 (与 linalg.generic 的逻辑等价)
// ============================================================================

/**
 * @brief 执行 Softmax 操作，分解为4个阶段。
 *        精确模拟 MLIR 的 linalg.generic 序列。
 * @param input_data  Input Matrix (BATCH_SIZE x FEATURES_SIZE)
 * @param output_data Output Matrix (BATCH_SIZE x FEATURES_SIZE)
 */
void kernel_softmax(const float* input_data, float* output_data) {
    // 为中间结果分配内存 (模拟 memref.alloc)
    std::vector<float> max_vals(BATCH_SIZE);
    std::vector<float> exp_vals(BATCH_SIZE * FEATURES_SIZE);
    std::vector<float> sum_vals(BATCH_SIZE);

    // --- STAGE 1: 求最大值 (per batch element) ---
    // linalg.fill ins(%f_min : f32) outs(%max_vals : memref<16xf32>)
    // std::numeric_limits<float>::lowest() is a very small negative number
    std::fill(max_vals.begin(), max_vals.end(), std::numeric_limits<float>::lowest());

    // Loop over (b, f) where b is parallel, f is reduction for max
    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        for (size_t f = 0; f < FEATURES_SIZE; ++f) {
            // max_vals(b) = max(in(b,f), max_vals(b))
            size_t in_idx = b * FEATURES_SIZE + f;
            max_vals[b] = std::max(max_vals[b], input_data[in_idx]);
        }
    }

    // --- STAGE 2: 减去最大值并求指数 ---
    // Loop over (b, f) where both are parallel
    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        for (size_t f = 0; f < FEATURES_SIZE; ++f) {
            // exp_vals(b,f) = exp(in(b,f) - max_vals(b))
            size_t in_idx = b * FEATURES_SIZE + f;
            size_t exp_idx = b * FEATURES_SIZE + f;
            exp_vals[exp_idx] = std::exp(input_data[in_idx] - max_vals[b]);
        }
    }

    // --- STAGE 3: 求和 (per batch element) ---
    // linalg.fill ins(%f_zero : f32) outs(%sum_vals : memref<16xf32>)
    std::fill(sum_vals.begin(), sum_vals.end(), 0.0f);

    // Loop over (b, f) where b is parallel, f is reduction for sum
    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        for (size_t f = 0; f < FEATURES_SIZE; ++f) {
            // sum_vals(b) = sum(exp_vals(b,f), sum_vals(b))
            size_t exp_idx = b * FEATURES_SIZE + f;
            sum_vals[b] += exp_vals[exp_idx];
        }
    }

    // --- STAGE 4: 除以和 ---
    // Loop over (b, f) where both are parallel
    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        for (size_t f = 0; f < FEATURES_SIZE; ++f) {
            // output_data(b,f) = exp_vals(b,f) / sum_vals(b)
            size_t exp_idx = b * FEATURES_SIZE + f;
            size_t out_idx = b * FEATURES_SIZE + f;
            output_data[out_idx] = exp_vals[exp_idx] / sum_vals[b];
        }
    }

    // 中间内存将自动释放 (std::vector 的析构函数)
}


// ============================================================================
// 主函数 (驱动整个流程)
// ============================================================================
int main() {
    // 1. 内存分配
    std::vector<float> input(BATCH_SIZE * FEATURES_SIZE);
    std::vector<float> output(BATCH_SIZE * FEATURES_SIZE);

    // 2. 数据初始化 (与 memref.global 的值匹配)
    // INPUT: dense<1.0>
    std::fill(input.begin(), input.end(), 1.0f);
    // OUTPUT: dense<0.0> (会被计算覆盖)
    std::fill(output.begin(), output.end(), 0.0f);

    // 3. 执行内核并计时
    double t_start = rtclock();
    kernel_softmax(input.data(), output.data());
    double t_end = rtclock();

    // 4. 输出结果与时间
    print_matrix_for_comparison(output.data(), BATCH_SIZE, FEATURES_SIZE);
    double time_s = t_end - t_start;
    printf("%f\n", time_s);

    return 0;
}