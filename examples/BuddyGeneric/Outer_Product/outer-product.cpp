#include <iostream>
#include <vector>
#include <chrono>
#include <numeric> // For std::fill
#include <cstddef> // For size_t
#include <cstdio>  // For printf

// ============================================================================
// 数据与维度定义 (与 MLIR 版本保持一致)
// ============================================================================
constexpr size_t I_DIM = 256;
constexpr size_t J_DIM = 512;

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
 * @brief 执行外积操作: C(i,j) = A(i) * B(j)
 *        模拟 linalg.generic 的两个嵌套循环。
 * @param A Input Vector (size I_DIM)
 * @param B Input Vector (size J_DIM)
 * @param C Output Matrix (I_DIM x J_DIM)
 */
void kernel_outer_product(const float* A, const float* B, float* C) {
    // iterator_types: ["parallel", "parallel"]
    // 对应 i, j 两个循环。
    for (size_t i = 0; i < I_DIM; ++i) {
        for (size_t j = 0; j < J_DIM; ++j) {
            // indexing_maps:
            // A -> (i)   => A[i * 1]
            // B -> (j)   => B[j * 1]
            // C -> (i, j) => C[i * J_DIM + j]
            
            // linalg.generic 的 outs 是原地更新 (+= semantics for reduction, but = for element-wise)
            // 对于外积 C(i,j) = A(i) * B(j)，它是一个赋值操作，而不是累加。
            // MLIR 的 linalg.generic 对于元素操作默认就是赋值 semantics。
            // 所以 C 的初始值 (0.0) 不会影响最终结果。
            C[i * J_DIM + j] = A[i] * B[j];
        }
    }
}


// ============================================================================
// 主函数 (驱动整个流程)
// ============================================================================
int main() {
    // 1. 内存分配
    std::vector<float> A(I_DIM);
    std::vector<float> B(J_DIM);
    std::vector<float> C(I_DIM * J_DIM);

    // 2. 数据初始化 (与 memref.global 的值匹配)
    // A: dense<2.0>
    std::fill(A.begin(), A.end(), 2.0f);
    // B: dense<3.0>
    std::fill(B.begin(), B.end(), 3.0f);
    // C: dense<0.0>
    std::fill(C.begin(), C.end(), 0.0f);

    // 3. 执行内核并计时
    double t_start = rtclock();
    kernel_outer_product(A.data(), B.data(), C.data());
    double t_end = rtclock();

    // 4. 输出结果与时间
    print_matrix_for_comparison(C.data(), I_DIM, J_DIM); // C 是 I x J 的矩阵
    double time_s = t_end - t_start;
    printf("%f\n", time_s);

    return 0;
}