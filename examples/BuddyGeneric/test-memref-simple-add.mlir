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
// 测试用例: 简单元素级加法 (memref) - 测试基本的并行维度识别
// 
// 这个测试用例验证：
// 1. memref类型的硬件能力分析
// 2. 内存布局的连续性分析
// 3. 直接内存操作的向量化可行性
// 4. 内存管理与向量化的结合
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

func.func @main() {
  call @test_simple_add_memref() : () -> ()
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1024] strides = [1] data =
// CHECK-NEXT: [5{{(, 5)*}}] 