#include <iostream>
#include <vector>
#include <chrono>
#include <numeric> // 用于 std::fill
#include <cstddef> // 用于 size_t
#include <cstdio>  // 用于 printf

// ============================================================================
// 数据与维度定义 (与 MLIR 版本保持一致)
// ============================================================================
constexpr size_t B_DIM = 4;   // Batch dimension
constexpr size_t M_DIM = 64;  // Matrix row dimension
constexpr size_t N_DIM = 128; // Matrix column dimension
constexpr size_t K_DIM = 256; // Reduction dimension

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 获取高精度时间戳。
 */
double rtclock() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

/**
 * @brief 以嵌套列表格式打印完整的3D张量，不进行任何截断。
 * 格式设计用于自动化测试脚本的精确匹配。
 * @param data 指向张量数据的指针。
 * @param d0, d1, d2 维度的实际大小 (Batch, M, N)。
 */
void print_3d_tensor_full(const float* data, size_t d0, size_t d1, size_t d2) {
    printf("["); // 整个张量的起始括号
    for (size_t b = 0; b < d0; ++b) {
        if (b > 0) {
            printf(",\n "); // Batch之间的分隔符
        }
        printf("["); // 矩阵的起始括号
        for (size_t m = 0; m < d1; ++m) {
            if (m > 0) {
                printf(",\n   "); // 行之间的分隔符，带缩进
            }
            printf("["); // 行的起始括号
            for (size_t n = 0; n < d2; ++n) {
                if (n > 0) {
                    printf(", "); // 元素之间的分隔符
                }
                printf("%g", data[b * (d1 * d2) + m * d2 + n]);
            }
            printf("]"); // 行的结束括号
        }
        printf("]"); // 矩阵的结束括号
    }
    printf("]\n"); // 整个张量的结束括号

    // 打印第一个元素用于正确性验证
    // printf("First element (Correctness Check): %g\n", data[0]);
}

// ============================================================================
// 内核函数 (与 linalg.generic 的逻辑等价)
// ============================================================================

void kernel_batch_gemm(const float* A, const float* B, float* C) {
    for (size_t b = 0; b < B_DIM; ++b) {
        for (size_t m = 0; m < M_DIM; ++m) {
            for (size_t n = 0; n < N_DIM; ++n) {
                for (size_t k = 0; k < K_DIM; ++k) {
                    size_t a_idx = b * (M_DIM * K_DIM) + m * K_DIM + k;
                    size_t b_idx = b * (K_DIM * N_DIM) + k * N_DIM + n;
                    size_t c_idx = b * (M_DIM * N_DIM) + m * N_DIM + n;
                    C[c_idx] += A[a_idx] * B[b_idx];
                }
            }
        }
    }
}

// ============================================================================
// 主函数 (驱动整个流程)
// ============================================================================
int main() {
    std::vector<float> A(B_DIM * M_DIM * K_DIM);
    std::vector<float> B(B_DIM * K_DIM * N_DIM);
    std::vector<float> C(B_DIM * M_DIM * N_DIM);

    std::fill(A.begin(), A.end(), 1.0f);
    std::fill(B.begin(), B.end(), 2.0f);
    std::fill(C.begin(), C.end(), 0.0f);

    double t_start = rtclock();
    kernel_batch_gemm(A.data(), B.data(), C.data());
    double t_end = rtclock();

    // 使用新的、打印完整内容的函数
    print_3d_tensor_full(C.data(), B_DIM, M_DIM, N_DIM);
    
    double time_s = t_end - t_start;
    printf("%f\n", time_s);

    return 0;
}