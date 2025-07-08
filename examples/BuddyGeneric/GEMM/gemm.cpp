#include <iostream>
#include <vector>
#include <chrono>
#include <numeric> // For std::fill
#include <cstddef> // For size_t
#include <cstdio>  // For printf

// ============================================================================
// 数据与维度定义 (与 MLIR 版本保持一致)
// ============================================================================
constexpr size_t M = 128;
constexpr size_t K = 512;
constexpr size_t N = 256;

// ============================================================================
// 辅助函数 (模拟 MLIR 的外部函数)
// ============================================================================

/**
 * @brief 模拟 MLIR 的 rtclock() 函数。
 */
double rtclock() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

/**
 * @brief 模拟 MLIR 测试脚本的打印格式。
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
            printf("%g", data[r * cols + c]); // Use %g to avoid trailing zeros
            if (c < cols - 1) {
                printf(",   ");
            }
        }
    }
    printf("]]\n");
}


// ============================================================================
// 内核函数 (与 linalg.generic 的逻辑等价)
// ============================================================================

/**
 * @brief 执行 GEMM 操作: C(m,n) += A(m,k) * B(k,n)
 */
void kernel_gemm(const float* A, const float* B, float* C) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}


// ============================================================================
// 主函数 (驱动整个流程)
// ============================================================================
int main() {
    // 1. 内存分配
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    // 2. 数据初始化
    std::fill(A.begin(), A.end(), 2.0f);
    std::fill(B.begin(), B.end(), 3.0f);
    std::fill(C.begin(), C.end(), 1.0f);

    // 3. 执行内核并计时
    double t_start = rtclock();
    kernel_gemm(A.data(), B.data(), C.data());
    double t_end = rtclock();

    // 4. 输出结果与时间 (使用新的打印函数)
    print_matrix_for_comparison(C.data(), M, N);
    double time_s = t_end - t_start;
    printf("%f\n", time_s);

    return 0;
}