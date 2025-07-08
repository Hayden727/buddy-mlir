#include <iostream>
#include <vector>
#include <chrono>
#include <numeric> // For std::fill
#include <cstddef> // For size_t
#include <cstdio>  // For printf

// ============================================================================
// 数据与维度定义 (与 MLIR 版本保持一致)
// ============================================================================
constexpr size_t SIZE = 4096; // 向量的长度

// ============================================================================
// 辅助函数 (与 GEMM/Conv2D 版本相同)
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
 * @brief 打印一维向量，以匹配MLIR测试脚本的输出格式。
 *        格式: [[val0, val1, ..., valN]]
 * @param data 指向向量数据的指针。
 * @param size 向量的长度。
 */
void print_vector_for_comparison(const float* data, size_t size) {
    printf("[[");
    for (size_t i = 0; i < size; ++i) {
        printf("%g", data[i]); // Use %g for clean float printing
        if (i < size - 1) {
            printf(", "); // Space after comma
        }
    }
    printf("]]\n");
}


// ============================================================================
// 内核函数 (与 linalg.generic 的逻辑等价)
// ============================================================================

/**
 * @brief 执行 AXPY 操作: Y(i) = a * X(i) + Y(i)
 *        精确模拟 linalg.generic 的单循环和区域计算。
 * @param a 标量乘数
 * @param X 输入向量
 * @param Y 输入/输出向量
 */
void kernel_axpy(float a, const float* X, float* Y) {
    // iterator_types: ["parallel"]
    for (size_t i = 0; i < SIZE; ++i) {
        // indexing_maps:
        // X -> (i)
        // Y -> (i)
        // 区域内的计算: Y(i) = a * X(i) + Y(i)
        Y[i] = a * X[i] + Y[i];
    }
}


// ============================================================================
// 主函数 (驱动整个流程)
// ============================================================================
int main() {
    // 1. 内存分配
    std::vector<float> X(SIZE);
    std::vector<float> Y(SIZE);

    // 2. 数据初始化 (与 memref.global 的值匹配)
    // X: dense<3.0>
    std::fill(X.begin(), X.end(), 3.0f);
    // Y: dense<10.0>
    std::fill(Y.begin(), Y.end(), 10.0f);

    // 标量 a
    float a_scalar = 2.0f; // 与 MLIR @main 中 arith.constant 2.0 : f32 匹配

    // 3. 执行内核并计时
    double t_start = rtclock();
    kernel_axpy(a_scalar, X.data(), Y.data());
    double t_end = rtclock();

    // 4. 输出结果与时间 (使用 print_vector_for_comparison)
    print_vector_for_comparison(Y.data(), SIZE);
    double time_s = t_end - t_start;
    printf("%f\n", time_s);

    return 0;
}