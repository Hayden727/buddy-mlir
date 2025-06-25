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
// 测试用例: 复杂的广播操作 - 测试内存访问模式
// 
// 这个测试用例验证：
// 1. 广播内存访问模式的识别
// 2. 不同维度输入的处理
// 3. 索引映射的复杂分析
// 4. 广播场景下的向量化可行性
// ===----------------------------------------------------------------------===//

func.func @test_broadcast_add() -> tensor<64x128xf32> {
  %t_start = call @rtclock() : () -> f64
  
  // 创建输入数据
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %input1 = tensor.splat %c2 : tensor<64x128xf32>    // 2D矩阵
  %input2 = tensor.splat %c3 : tensor<128xf32>       // 1D向量，将被广播
  %output_empty = tensor.empty() : tensor<64x128xf32>
  
  // 执行广播加法：matrix[i][j] + vector[j]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,  // 矩阵：完整访问
                     affine_map<(d0, d1) -> (d1)>,       // 向量：只访问第二维
                     affine_map<(d0, d1) -> (d0, d1)>],  // 输出：完整访问
    iterator_types = ["parallel", "parallel"]
  } ins(%input1, %input2 : tensor<64x128xf32>, tensor<128xf32>) 
    outs(%output_empty : tensor<64x128xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32
  } -> tensor<64x128xf32>
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (2.0 + 3.0 = 5.0)
  %tensor_unranked = tensor.cast %result : tensor<64x128xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64
  
  return %result : tensor<64x128xf32>
}

func.func @main() {
  %result = call @test_broadcast_add() : () -> tensor<64x128xf32>
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [64, 128] strides = [128, 1] data =
// CHECK-NEXT: [
// CHECK-SAME: [5{{(, 5)*}}], 