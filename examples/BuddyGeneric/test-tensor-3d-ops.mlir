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
// 测试用例: 三维张量操作 - 测试高维度分析
// 
// 这个测试用例验证：
// 1. 高维度张量的处理能力
// 2. 三维并行迭代器的识别
// 3. 复杂数学运算的分析（指数函数）
// 4. 高维度场景下的向量化决策
// ===----------------------------------------------------------------------===//

func.func @test_tensor_3d_ops() -> tensor<32x64x128xf32> {
  %t_start = call @rtclock() : () -> f64
  
  // 创建三维输入张量
  %c1 = arith.constant 1.0 : f32
  %input = tensor.splat %c1 : tensor<32x64x128xf32>
  %output_empty = tensor.empty() : tensor<32x64x128xf32>
  
  // 执行三维张量的复杂数学运算：exp(input * 2.0)
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<32x64x128xf32>) 
    outs(%output_empty : tensor<32x64x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c2 = arith.constant 2.0 : f32
    %mul = arith.mulf %in, %c2 : f32
    %exp = math.exp %mul : f32
    linalg.yield %exp : f32
  } -> tensor<32x64x128xf32>
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (exp(1.0 * 2.0) = exp(2.0) ≈ 7.389)
  %tensor_unranked = tensor.cast %result : tensor<32x64x128xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64
  
  return %result : tensor<32x64x128xf32>
}

func.func @main() {
  %result = call @test_tensor_3d_ops() : () -> tensor<32x64x128xf32>
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [32, 64, 128] strides = [8192, 128, 1] data =
// CHECK-NEXT: [
// CHECK-SAME: [
// CHECK-SAME: [7.{{[0-9]+}}{{(, 7\.[0-9]+)*}}],
 