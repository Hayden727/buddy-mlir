#include <iostream>
#include <vector>
#include <chrono>
#include <numeric> // For std::fill
#include <cstddef> // For size_t
#include <cstdio>  // For printf

// ============================================================================
// 数据与维度定义 (与 MLIR 版本保持一致)
// ============================================================================
constexpr size_t B = 16;  // Batch dimension
constexpr size_t I = 128; // Output matrix rows
constexpr size_t K = 256; // Reduction dimension / Inner dimension

// ============================================================================
// 辅助函数 (与 GEMM/AXPY 版本相同)
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
 * @brief 执行批量矩阵-向量乘操作: C(b,i) += A(b,i,k) * B(b,k)
 *        精确模拟 linalg.generic 的三个嵌套循环和区域计算。
 * @param A 3D Batch Matrix Tensor (B x I x K)
 * @param B 2D Batch Vector Tensor (B x K)
 * @param C 2D Batch Output Tensor (B x I)
 */
void kernel_batch_mat_vec(const float* A, const float* B_data, float* C) {
    // iterator_types = ["parallel", "parallel", "reduction"]
    // 对应 b, i, k 三个循环。
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < I; ++i) {
            for (size_t k = 0; k < K; ++k) {
                // indexing_maps:
                // A -> (b, i, k)  => A[b * I * K + i * K + k]
                // B -> (b, k)    => B[b * K + k]
                // C -> (b, i)    => C[b * I + i]
                C[b * I + i] += A[b * I * K + i * K + k] * B_data[b * K + k];
            }
        }
    }
}


// ============================================================================
// 主函数 (驱动整个流程)
// ============================================================================
int main() {
    // 1. 内存分配
    std::vector<float> A(B * I * K);
    std::vector<float> B_vec(B * K);
    std::vector<float> C(B * I);

    // 2. 数据初始化 (与 memref.global 的值匹配)
    // A: dense<1.0>
    std::fill(A.begin(), A.end(), 1.0f);
    // B: dense<2.0>
    std::fill(B_vec.begin(), B_vec.end(), 2.0f);
    // C: dense<0.0>
    std::fill(C.begin(), C.end(), 0.0f);

    // 3. 执行内核并计时
    double t_start = rtclock();
    kernel_batch_mat_vec(A.data(), B_vec.data(), C.data());
    double t_end = rtclock();

    // 4. 输出结果与时间
    print_matrix_for_comparison(C.data(), B, I); // C 是 B x I 的矩阵
    double time_s = t_end - t_start;
    printf("%f\n", time_s);

    return 0;
}