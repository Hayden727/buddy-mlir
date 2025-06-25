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
// 测试用例: 二维矩阵元素级操作 - 测试多维并行
// 
// 这个测试用例验证：
// 1. 多维并行迭代器的正确识别和处理
// 2. 二维内存访问模式的分析
// 3. 多维向量化的可行性判断
// 4. 复杂计算表达式的处理（乘法+加法）
// ===----------------------------------------------------------------------===//

func.func @test_matrix_elementwise() -> tensor<256x512xf32> {
  %t_start = call @rtclock() : () -> f64
  
  // 创建输入数据
  %c4 = arith.constant 4.0 : f32
  %c2 = arith.constant 2.0 : f32
  %input1 = tensor.splat %c4 : tensor<256x512xf32>
  %input2 = tensor.splat %c2 : tensor<256x512xf32>
  %output_empty = tensor.empty() : tensor<256x512xf32>
  
  // 执行二维矩阵的元素级操作：(a * b + 1)
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input1, %input2 : tensor<256x512xf32>, tensor<256x512xf32>) 
    outs(%output_empty : tensor<256x512xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %mul = arith.mulf %in0, %in1 : f32
    %c1 = arith.constant 1.0 : f32
    %add = arith.addf %mul, %c1 : f32
    linalg.yield %add : f32
  } -> tensor<256x512xf32>
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (4.0 * 2.0 + 1.0 = 9.0)
  %tensor_unranked = tensor.cast %result : tensor<256x512xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64
  
  return %result : tensor<256x512xf32>
}

func.func @main() {
  %result = call @test_matrix_elementwise() : () -> tensor<256x512xf32>
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [256, 512] strides = [512, 1] data =
// CHECK-NEXT: [
// CHECK-SAME: [9{{(, 9)*}}], 