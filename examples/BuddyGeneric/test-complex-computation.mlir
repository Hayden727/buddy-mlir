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
// 测试用例: 多输入复杂计算 - 测试操作复杂度分析
// 
// 这个测试用例验证：
// 1. 多个输入操作数的处理
// 2. 复杂计算表达式的分析（乘法+加法+开方+除法）
// 3. 计算密集型操作的向量化可行性
// 4. 多操作数场景下的内存访问分析
// ===----------------------------------------------------------------------===//

func.func @test_complex_computation() -> tensor<512xf32> {
  %t_start = call @rtclock() : () -> f64
  
  // 创建三个输入向量
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %c4 = arith.constant 4.0 : f32
  %input1 = tensor.splat %c2 : tensor<512xf32>
  %input2 = tensor.splat %c3 : tensor<512xf32>
  %input3 = tensor.splat %c4 : tensor<512xf32>
  %output_empty = tensor.empty() : tensor<512xf32>
  
  // 执行复杂计算：sqrt(a * b + c) / 1.0
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input1, %input2, %input3 : tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) 
    outs(%output_empty : tensor<512xf32>) {
  ^bb0(%in0: f32, %in1: f32, %in2: f32, %out: f32):
    %mul1 = arith.mulf %in0, %in1 : f32
    %add = arith.addf %mul1, %in2 : f32
    %sqrt = math.sqrt %add : f32
    %c1 = arith.constant 1.0 : f32
    %div = arith.divf %sqrt, %c1 : f32
    linalg.yield %div : f32
  } -> tensor<512xf32>
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (sqrt(2.0 * 3.0 + 4.0) / 1.0 = sqrt(10.0) ≈ 3.162)
  %tensor_unranked = tensor.cast %result : tensor<512xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64
  
  return %result : tensor<512xf32>
}

func.func @main() {
  %result = call @test_complex_computation() : () -> tensor<512xf32>
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [512] strides = [1] data =
// CHECK-NEXT: [3.{{[0-9]+}}{{(, 3\.[0-9]+)*}}] 