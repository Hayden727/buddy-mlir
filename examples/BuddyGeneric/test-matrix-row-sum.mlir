// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
// RUN: | buddy-opt \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -tensor-bufferize \
// RUN:     -buffer-deallocation \
// RUN:     -finalizing-bufferize \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

// ===----------------------------------------------------------------------===//
// 测试用例: 矩阵行求和 - 测试混合迭代器类型
// 
// 这个测试用例验证：
// 1. 混合迭代器类型的正确识别（parallel + reduction）
// 2. 部分维度reduction的处理
// 3. 输出维度与输入维度不匹配的情况
// 4. 混合场景下的向量化决策
// ===----------------------------------------------------------------------===//

func.func @test_matrix_row_sum() -> tensor<128xf32> {
  %t_start = call @rtclock() : () -> f64
  
  // 创建输入矩阵，每个元素都是1.0
  %c1 = arith.constant 1.0 : f32
  %input = tensor.splat %c1 : tensor<128x256xf32>
  
  // 初始化输出向量为0
  %c0 = arith.constant 0.0 : f32
  %init = tensor.splat %c0 : tensor<128xf32>
  
  // 执行矩阵行求和：output[i] = sum(input[i][j] for j in range(256))
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<128x256xf32>) 
    outs(%init : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %out, %in : f32
    linalg.yield %add : f32
  } -> tensor<128xf32>
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (每行求和: 1.0 * 256 = 256.0)
  %tensor_unranked = tensor.cast %result : tensor<128xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64
  
  return %result : tensor<128xf32>
}

func.func @main() {
  %result = call @test_matrix_row_sum() : () -> tensor<128xf32>
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [128] strides = [1] data =
// CHECK-NEXT: [256{{(, 256)*}}] 