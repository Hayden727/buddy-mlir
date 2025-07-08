// ============================================================================
// GEMM C++ - 大规模 (Large) - 未优化
// 对应 MLIR 测试用例: M=256, N=512, K=1024
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric> // For std::fill
#include <cstddef> // For size_t
#include <cstdio>  // For printf

// ============================================================================
// 数据与维度定义 (大规模 / Large)
// ============================================================================
constexpr size_t M = 256;
constexpr size_t K = 1024;
constexpr size_t N = 512;

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
 * @brief 模拟 MLIR 测试脚本的打印格式，带有截断功能。
 */
void print_matrix_for_comparison(const float* data, size_t rows, size_t cols) {
    const size_t max_print_rows = 16;
    const size_t max_print_cols = 16;
    bool truncated = (rows > max_print_rows) || (cols > max_print_cols);
    
    size_t print_rows = (rows > max_print_rows) ? max_print_rows : rows;
    size_t print_cols = (cols > max_print_cols) ? max_print_cols : cols;

    printf("[[");
    for (size_t r = 0; r < print_rows; ++r) {
        if (r > 0) {
            printf("], \n [");
        } else {
            printf("[");
        }
        for (size_t c = 0; c < print_cols; ++c) {
            printf("%g", data[r * N + c]); // 使用 N 作为列步长
            if (c < print_cols - 1) {
                printf(",   ");
            }
        }
    }
    if (truncated) {
        printf("... (truncated)");
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

    // 4. 输出结果与时间
    print_matrix_for_comparison(C.data(), M, N);
    double time_s = t_end - t_start;
    printf("%f\n", time_s);

    return 0;
}