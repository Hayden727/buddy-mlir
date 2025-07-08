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
// 辅助函数 (修正版)
// ============================================================================

double rtclock() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

/**
 * @brief 打印单个标量值，以匹配MLIR测试脚本的输出格式。
 *        格式: [value]
 * @param value 要打印的标量。
 */
void print_scalar_for_comparison(float value) {
    // FIX: 移除一层括号
    printf("[%g]\n", value); // Use %g for clean float printing
}


// ============================================================================
// 内核函数 (与 linalg.generic 的逻辑等价)
// ============================================================================

/**
 * @brief 执行点积操作: c += A(i) * B(i)
 */
void kernel_dot_product(const float* A, const float* B, float& c) { // c 使用引用以便原地更新
    for (size_t i = 0; i < SIZE; ++i) {
        c += A[i] * B[i];
    }
}


// ============================================================================
// 主函数 (驱动整个流程)
// ============================================================================
int main() {
    // 1. 内存分配
    std::vector<float> A(SIZE);
    std::vector<float> B(SIZE);
    float C_scalar; // 声明一个 float 变量用于标量 C

    // 2. 数据初始化 (与 memref.global 的值匹配)
    std::fill(A.begin(), A.end(), 2.0f);
    std::fill(B.begin(), B.end(), 3.0f);
    C_scalar = 0.0f; // 初始化标量 C

    // 3. 执行内核并计时
    double t_start = rtclock();
    kernel_dot_product(A.data(), B.data(), C_scalar);
    double t_end = rtclock();

    // 4. 输出结果与时间
    print_scalar_for_comparison(C_scalar);
    double time_s = t_end - t_start;
    printf("%f\n", time_s);

    return 0;
}