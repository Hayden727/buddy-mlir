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
func.func private @printMemrefF64(%ptr : tensor<*xf64>)

// ===----------------------------------------------------------------------===//
// 测试用例1: 简单元素级加法 - 测试基本的并行维度识别
// ===----------------------------------------------------------------------===//

func.func @test_simple_add() -> tensor<1024xf32> {
  %t_start = call @rtclock() : () -> f64
  
  // 创建输入数据
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %input1 = tensor.splat %c2 : tensor<1024xf32>
  %input2 = tensor.splat %c3 : tensor<1024xf32>
  %output_empty = tensor.empty() : tensor<1024xf32>
  
  // 执行linalg.generic操作
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

// ===----------------------------------------------------------------------===//
// 测试用例2: 二维矩阵元素级操作 - 测试多维并行
// ===----------------------------------------------------------------------===//

func.func @test_matrix_elementwise() -> tensor<256x512xf32> {
  %t_start = call @rtclock() : () -> f64
  
  %c4 = arith.constant 4.0 : f32
  %c2 = arith.constant 2.0 : f32
  %input1 = tensor.splat %c4 : tensor<256x512xf32>
  %input2 = tensor.splat %c2 : tensor<256x512xf32>
  %output_empty = tensor.empty() : tensor<256x512xf32>
  
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

// ===----------------------------------------------------------------------===//
// 测试用例3: 向量内积 - 测试reduction操作
// ===----------------------------------------------------------------------===//

func.func @test_vector_dot_product() -> tensor<f32> {
  %t_start = call @rtclock() : () -> f64
  
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %input1 = tensor.splat %c2 : tensor<1024xf32>
  %input2 = tensor.splat %c3 : tensor<1024xf32>
  
  %c0 = arith.constant 0.0 : f32
  %init = tensor.from_elements %c0 : tensor<f32>
  
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]
  } ins(%input1, %input2 : tensor<1024xf32>, tensor<1024xf32>) 
    outs(%init : tensor<f32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %mul = arith.mulf %in0, %in1 : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<f32>
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (2.0 * 3.0 * 1024 = 6144.0)
  %tensor_unranked = tensor.cast %result : tensor<f32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64
  
  return %result : tensor<f32>
}

// ===----------------------------------------------------------------------===//
// 测试用例4: 矩阵行求和 - 测试混合迭代器类型
// ===----------------------------------------------------------------------===//

func.func @test_matrix_row_sum() -> tensor<128xf32> {
  %t_start = call @rtclock() : () -> f64
  
  %c1 = arith.constant 1.0 : f32
  %input = tensor.splat %c1 : tensor<128x256xf32>
  
  %c0 = arith.constant 0.0 : f32
  %init = tensor.splat %c0 : tensor<128xf32>
  
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

// ===----------------------------------------------------------------------===//
// 测试用例5: 复杂的广播操作 - 测试内存访问模式
// ===----------------------------------------------------------------------===//

func.func @test_broadcast_add() -> tensor<64x128xf32> {
  %t_start = call @rtclock() : () -> f64
  
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %input1 = tensor.splat %c2 : tensor<64x128xf32>
  %input2 = tensor.splat %c3 : tensor<128xf32>
  %output_empty = tensor.empty() : tensor<64x128xf32>
  
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
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

// ===----------------------------------------------------------------------===//
// 测试用例6: 三维张量操作 - 测试高维度分析
// ===----------------------------------------------------------------------===//

func.func @test_tensor_3d_ops() -> tensor<32x64x128xf32> {
  %t_start = call @rtclock() : () -> f64
  
  %c1 = arith.constant 1.0 : f32
  %input = tensor.splat %c1 : tensor<32x64x128xf32>
  %output_empty = tensor.empty() : tensor<32x64x128xf32>
  
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

// ===----------------------------------------------------------------------===//
// 测试用例7: 多输入复杂计算 - 测试操作复杂度分析
// ===----------------------------------------------------------------------===//

func.func @test_complex_computation() -> tensor<512xf32> {
  %t_start = call @rtclock() : () -> f64
  
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %c4 = arith.constant 4.0 : f32
  %input1 = tensor.splat %c2 : tensor<512xf32>
  %input2 = tensor.splat %c3 : tensor<512xf32>
  %input3 = tensor.splat %c4 : tensor<512xf32>
  %output_empty = tensor.empty() : tensor<512xf32>
  
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

// ===----------------------------------------------------------------------===//
// 主测试函数
// ===----------------------------------------------------------------------===//

func.func @main() {
  // 打印测试开始信息
  %test_start = call @rtclock() : () -> f64
  
  // 执行各个测试用例
  %result1 = call @test_simple_add() : () -> tensor<1024xf32>
  %result2 = call @test_matrix_elementwise() : () -> tensor<256x512xf32>
  %result3 = call @test_vector_dot_product() : () -> tensor<f32>
  %result4 = call @test_matrix_row_sum() : () -> tensor<128xf32>
  %result5 = call @test_broadcast_add() : () -> tensor<64x128xf32>
  %result6 = call @test_tensor_3d_ops() : () -> tensor<32x64x128xf32>
  %result7 = call @test_complex_computation() : () -> tensor<512xf32>
  
  %test_end = call @rtclock() : () -> f64
  %total_time = arith.subf %test_end, %test_start : f64
  
  // 打印总测试时间
  vector.print %total_time : f64
  
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1024] strides = [1] data =
// CHECK-NEXT: [5{{(, 5)*}}]

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [256, 512] strides = [512, 1] data =
// CHECK-NEXT: [
// CHECK-SAME: [9{{(, 9)*}}],

// CHECK: Unranked Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
// CHECK-NEXT: [6144]

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [128] strides = [1] data =
// CHECK-NEXT: [256{{(, 256)*}}]

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [64, 128] strides = [128, 1] data =
// CHECK-NEXT: [
// CHECK-SAME: [5{{(, 5)*}}],

// CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [32, 64, 128] strides = [8192, 128, 1] data =
// CHECK-NEXT: [
// CHECK-SAME: [
// CHECK-SAME: [7.{{[0-9]+}}{{(, 7\.[0-9]+)*}}],

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [512] strides = [1] data =
// CHECK-NEXT: [3.{{[0-9]+}}{{(, 3\.[0-9]+)*}}] 