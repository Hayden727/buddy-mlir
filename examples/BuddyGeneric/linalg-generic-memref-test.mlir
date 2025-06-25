// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
// RUN: | buddy-opt \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -buffer-deallocation \
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
func.func private @printMemrefF32(%ptr : memref<*xf32>)

// ===----------------------------------------------------------------------===//
// 测试用例1: 简单元素级加法 (memref) - 测试基本的并行维度识别
// ===----------------------------------------------------------------------===//

func.func @test_simple_add_memref() {
  %t_start = call @rtclock() : () -> f64
  
  // 分配内存并初始化数据
  %input1 = memref.alloc() : memref<1024xf32>
  %input2 = memref.alloc() : memref<1024xf32>
  %output = memref.alloc() : memref<1024xf32>
  
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  linalg.fill ins(%c2 : f32) outs(%input1 : memref<1024xf32>)
  linalg.fill ins(%c3 : f32) outs(%input2 : memref<1024xf32>)
  
  // 执行linalg.generic操作
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input1, %input2 : memref<1024xf32>, memref<1024xf32>) 
    outs(%output : memref<1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (2.0 + 3.0 = 5.0)
  %output_unranked = memref.cast %output : memref<1024xf32> to memref<*xf32>
  call @printMemrefF32(%output_unranked) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  // 释放内存
  memref.dealloc %input1 : memref<1024xf32>
  memref.dealloc %input2 : memref<1024xf32>
  memref.dealloc %output : memref<1024xf32>
  
  return
}

// ===----------------------------------------------------------------------===//
// 测试用例2: 二维矩阵元素级操作 (memref) - 测试多维并行
// ===----------------------------------------------------------------------===//

func.func @test_matrix_elementwise_memref() {
  %t_start = call @rtclock() : () -> f64
  
  %input1 = memref.alloc() : memref<256x512xf32>
  %input2 = memref.alloc() : memref<256x512xf32>
  %output = memref.alloc() : memref<256x512xf32>
  
  %c4 = arith.constant 4.0 : f32
  %c2 = arith.constant 2.0 : f32
  linalg.fill ins(%c4 : f32) outs(%input1 : memref<256x512xf32>)
  linalg.fill ins(%c2 : f32) outs(%input2 : memref<256x512xf32>)
  
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input1, %input2 : memref<256x512xf32>, memref<256x512xf32>) 
    outs(%output : memref<256x512xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %mul = arith.mulf %in0, %in1 : f32
    %c1 = arith.constant 1.0 : f32
    %add = arith.addf %mul, %c1 : f32
    linalg.yield %add : f32
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (4.0 * 2.0 + 1.0 = 9.0)
  %output_unranked = memref.cast %output : memref<256x512xf32> to memref<*xf32>
  call @printMemrefF32(%output_unranked) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  memref.dealloc %input1 : memref<256x512xf32>
  memref.dealloc %input2 : memref<256x512xf32>
  memref.dealloc %output : memref<256x512xf32>
  
  return
}

// ===----------------------------------------------------------------------===//
// 测试用例3: 向量内积 (memref) - 测试reduction操作
// ===----------------------------------------------------------------------===//

func.func @test_vector_dot_product_memref() {
  %t_start = call @rtclock() : () -> f64
  
  %input1 = memref.alloc() : memref<1024xf32>
  %input2 = memref.alloc() : memref<1024xf32>
  %output = memref.alloc() : memref<f32>
  
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %c0 = arith.constant 0.0 : f32
  linalg.fill ins(%c2 : f32) outs(%input1 : memref<1024xf32>)
  linalg.fill ins(%c3 : f32) outs(%input2 : memref<1024xf32>)
  linalg.fill ins(%c0 : f32) outs(%output : memref<f32>)
  
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]
  } ins(%input1, %input2 : memref<1024xf32>, memref<1024xf32>) 
    outs(%output : memref<f32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %mul = arith.mulf %in0, %in1 : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (2.0 * 3.0 * 1024 = 6144.0)
  %output_unranked = memref.cast %output : memref<f32> to memref<*xf32>
  call @printMemrefF32(%output_unranked) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  memref.dealloc %input1 : memref<1024xf32>
  memref.dealloc %input2 : memref<1024xf32>
  memref.dealloc %output : memref<f32>
  
  return
}

// ===----------------------------------------------------------------------===//
// 测试用例4: 矩阵行求和 (memref) - 测试混合迭代器类型
// ===----------------------------------------------------------------------===//

func.func @test_matrix_row_sum_memref() {
  %t_start = call @rtclock() : () -> f64
  
  %input = memref.alloc() : memref<128x256xf32>
  %output = memref.alloc() : memref<128xf32>
  
  %c1 = arith.constant 1.0 : f32
  %c0 = arith.constant 0.0 : f32
  linalg.fill ins(%c1 : f32) outs(%input : memref<128x256xf32>)
  linalg.fill ins(%c0 : f32) outs(%output : memref<128xf32>)
  
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : memref<128x256xf32>) 
    outs(%output : memref<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %out, %in : f32
    linalg.yield %add : f32
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (每行求和: 1.0 * 256 = 256.0)
  %output_unranked = memref.cast %output : memref<128xf32> to memref<*xf32>
  call @printMemrefF32(%output_unranked) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  memref.dealloc %input : memref<128x256xf32>
  memref.dealloc %output : memref<128xf32>
  
  return
}

// ===----------------------------------------------------------------------===//
// 测试用例5: 复杂的广播操作 (memref) - 测试内存访问模式
// ===----------------------------------------------------------------------===//

func.func @test_broadcast_add_memref() {
  %t_start = call @rtclock() : () -> f64
  
  %input1 = memref.alloc() : memref<64x128xf32>
  %input2 = memref.alloc() : memref<128xf32>
  %output = memref.alloc() : memref<64x128xf32>
  
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  linalg.fill ins(%c2 : f32) outs(%input1 : memref<64x128xf32>)
  linalg.fill ins(%c3 : f32) outs(%input2 : memref<128xf32>)
  
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input1, %input2 : memref<64x128xf32>, memref<128xf32>) 
    outs(%output : memref<64x128xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (2.0 + 3.0 = 5.0)
  %output_unranked = memref.cast %output : memref<64x128xf32> to memref<*xf32>
  call @printMemrefF32(%output_unranked) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  memref.dealloc %input1 : memref<64x128xf32>
  memref.dealloc %input2 : memref<128xf32>
  memref.dealloc %output : memref<64x128xf32>
  
  return
}

// ===----------------------------------------------------------------------===//
// 测试用例6: 转置操作 (memref) - 测试非对角内存访问
// ===----------------------------------------------------------------------===//

func.func @test_transpose_memref() {
  %t_start = call @rtclock() : () -> f64
  
  %input = memref.alloc() : memref<256x512xf32>
  %output = memref.alloc() : memref<512x256xf32>
  
  %c7 = arith.constant 7.0 : f32
  linalg.fill ins(%c7 : f32) outs(%input : memref<256x512xf32>)
  
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d1, d0)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : memref<256x512xf32>) 
    outs(%output : memref<512x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (转置后数据应该保持7.0)
  %output_unranked = memref.cast %output : memref<512x256xf32> to memref<*xf32>
  call @printMemrefF32(%output_unranked) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  memref.dealloc %input : memref<256x512xf32>
  memref.dealloc %output : memref<512x256xf32>
  
  return
}

// ===----------------------------------------------------------------------===//
// 测试用例7: 动态维度测试 (memref) - 测试未知大小处理
// ===----------------------------------------------------------------------===//

func.func @test_dynamic_memref(%dim0: index, %dim1: index) {
  %t_start = call @rtclock() : () -> f64
  
  %input1 = memref.alloc(%dim0, %dim1) : memref<?x?xf32>
  %input2 = memref.alloc(%dim0, %dim1) : memref<?x?xf32>
  %output = memref.alloc(%dim0, %dim1) : memref<?x?xf32>
  
  %c6 = arith.constant 6.0 : f32
  %c4 = arith.constant 4.0 : f32
  linalg.fill ins(%c6 : f32) outs(%input1 : memref<?x?xf32>)
  linalg.fill ins(%c4 : f32) outs(%input2 : memref<?x?xf32>)
  
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input1, %input2 : memref<?x?xf32>, memref<?x?xf32>) 
    outs(%output : memref<?x?xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %sub = arith.subf %in0, %in1 : f32
    linalg.yield %sub : f32
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (6.0 - 4.0 = 2.0)
  %output_unranked = memref.cast %output : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%output_unranked) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  memref.dealloc %input1 : memref<?x?xf32>
  memref.dealloc %input2 : memref<?x?xf32>
  memref.dealloc %output : memref<?x?xf32>
  
  return
}

// ===----------------------------------------------------------------------===//
// 主测试函数
// ===----------------------------------------------------------------------===//

func.func @main() {
  // 打印测试开始信息
  %test_start = call @rtclock() : () -> f64
  
  // 执行各个测试用例
  call @test_simple_add_memref() : () -> ()
  call @test_matrix_elementwise_memref() : () -> ()
  call @test_vector_dot_product_memref() : () -> ()
  call @test_matrix_row_sum_memref() : () -> ()
  call @test_broadcast_add_memref() : () -> ()
  call @test_transpose_memref() : () -> ()
  
  // 动态维度测试
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  call @test_dynamic_memref(%c128, %c64) : (index, index) -> ()
  
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

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [512, 256] strides = [256, 1] data =
// CHECK-NEXT: [
// CHECK-SAME: [7{{(, 7)*}}],

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [128, 64] strides = [64, 1] data =
// CHECK-NEXT: [
// CHECK-SAME: [2{{(, 2)*}}], 