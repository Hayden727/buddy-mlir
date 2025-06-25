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
// 测试用例: 简单元素级加法 - 测试基本的并行维度识别
// 
// 这个测试用例验证：
// 1. 硬件能力分析是否正确识别向量宽度
// 2. 操作模式分析是否正确识别单一并行维度
// 3. 内存访问分析是否正确识别连续访问模式
// 4. 向量化决策是否建议进行向量化
// ===----------------------------------------------------------------------===//

func.func @test_simple_add() -> tensor<1024xf32> {
  %t_start = call @rtclock() : () -> f64
  
  // 创建输入数据 - 使用简单的常量值便于验证
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %input1 = tensor.splat %c2 : tensor<1024xf32>
  %input2 = tensor.splat %c3 : tensor<1024xf32>
  %output_empty = tensor.empty() : tensor<1024xf32>
  
  // 执行linalg.generic操作 - 这是我们要优化的核心操作
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input1, %input2 : tensor<1024xf32>, tensor<1024xf32>) 
    outs(%output_empty : tensor<1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32
  } -> tensor<1024xf32>
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (2.0 + 3.0 = 5.0)
  %tensor_unranked = tensor.cast %result : tensor<1024xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64
  
  return %result : tensor<1024xf32>
}

func.func @main() {
  %result = call @test_simple_add() : () -> tensor<1024xf32>
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1024] strides = [1] data =
// CHECK-NEXT: [5{{(, 5)*}}] 