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
// 测试用例: 动态维度测试 - 测试未知大小处理
// 
// 这个测试用例验证：
// 1. 动态维度的识别和处理
// 2. 运行时大小的向量化决策
// 3. 未知维度大小情况下的内存访问分析
// 4. 动态形状的硬件适配能力
// ===----------------------------------------------------------------------===//

func.func @test_dynamic_dims(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  %t_start = call @rtclock() : () -> f64
  
  // 执行动态维度的linalg.generic操作
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : memref<?xf32>, memref<?xf32>) 
    outs(%arg2 : memref<?xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %mul = arith.mulf %in0, %in1 : f32
    %c2 = arith.constant 2.0 : f32
    %add = arith.addf %mul, %c2 : f32
    linalg.yield %add : f32
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果
  %output_unranked = memref.cast %arg2 : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%output_unranked) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  return
}

func.func @main() {
  // 创建动态大小的memref (运行时大小为256)
  %c256 = arith.constant 256 : index
  %input1 = memref.alloc(%c256) : memref<?xf32>
  %input2 = memref.alloc(%c256) : memref<?xf32>
  %output = memref.alloc(%c256) : memref<?xf32>
  
  // 初始化数据
  %c3 = arith.constant 3.0 : f32
  %c4 = arith.constant 4.0 : f32
  linalg.fill ins(%c3 : f32) outs(%input1 : memref<?xf32>)
  linalg.fill ins(%c4 : f32) outs(%input2 : memref<?xf32>)
  
  // 调用测试函数
  call @test_dynamic_dims(%input1, %input2, %output) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  
  // 释放内存
  memref.dealloc %input1 : memref<?xf32>
  memref.dealloc %input2 : memref<?xf32>
  memref.dealloc %output : memref<?xf32>
  
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [256] strides = [1] data =
// CHECK-NEXT: [14{{(, 14)*}}] 