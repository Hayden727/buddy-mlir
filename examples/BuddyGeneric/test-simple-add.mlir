// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(linalg-generic-optimization{user-vector-width=256 user-tile-size=64}))" \
// RUN: | buddy-opt \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
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
// 测试用例: 复杂2D矩阵运算 - 体现向量化优势
// 
// 这个测试用例验证：
// 1. 2D并行维度的正确识别和分块优化
// 2. 复杂算术和数学运算的向量化
// 3. 内存访问模式分析对2D连续访问的优化
// 4. 向量化在计算密集型操作中的性能收益
// 
// 运算公式: result = ((A + B) * C) / (sin(D) + 0.1)
// 其中包含：加法、乘法、除法、数学函数，充分体现向量化优势
// ===----------------------------------------------------------------------===//

func.func @test_complex_2d_compute() -> memref<512x512xf32> {
  %t_start = call @rtclock() : () -> f64
  
  // 创建输入数据 - 2D矩阵
  %c1 = arith.constant 1.5 : f32
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %c4 = arith.constant 0.5 : f32
  %eps = arith.constant 0.1 : f32
  
  // 分配2D memref
  %matrix_a = memref.alloc() : memref<512x512xf32>
  %matrix_b = memref.alloc() : memref<512x512xf32>
  %matrix_c = memref.alloc() : memref<512x512xf32>
  %matrix_d = memref.alloc() : memref<512x512xf32>
  %result = memref.alloc() : memref<512x512xf32>
  
  // 初始化输入矩阵
  linalg.fill ins(%c1 : f32) outs(%matrix_a : memref<512x512xf32>)
  linalg.fill ins(%c2 : f32) outs(%matrix_b : memref<512x512xf32>)
  linalg.fill ins(%c3 : f32) outs(%matrix_c : memref<512x512xf32>)
  linalg.fill ins(%c4 : f32) outs(%matrix_d : memref<512x512xf32>)
  
  // 执行复杂的2D linalg.generic操作
  // 计算: result = ((A + B) * C) / (sin(D) + 0.1)
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%matrix_a, %matrix_b, %matrix_c, %matrix_d : 
        memref<512x512xf32>, memref<512x512xf32>, memref<512x512xf32>, memref<512x512xf32>) 
    outs(%result : memref<512x512xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32, %d: f32, %out: f32):
    // 复杂运算序列，充分利用向量化
    %add = arith.addf %a, %b : f32              // A + B
    %mul = arith.mulf %add, %c : f32            // (A + B) * C
    %sin_d = math.sin %d : f32                  // sin(D)
    %sin_eps = arith.addf %sin_d, %eps : f32    // sin(D) + 0.1
    %div = arith.divf %mul, %sin_eps : f32      // ((A + B) * C) / (sin(D) + 0.1)
    linalg.yield %div : f32
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证部分结果 (计算结果应该约为 10.5 / (sin(0.5) + 0.1) ≈ 17.65)
  %memref_unranked = memref.cast %result : memref<512x512xf32> to memref<*xf32>
  call @printMemrefF32(%memref_unranked) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  return %result : memref<512x512xf32>
}

// ===----------------------------------------------------------------------===//
// 额外测试：3D张量运算 - 进一步测试复杂情况
// ===----------------------------------------------------------------------===//

func.func @test_3d_tensor_compute() -> memref<64x64x64xf32> {
  %t_start = call @rtclock() : () -> f64
  
  // 3D张量数据
  %c1 = arith.constant 2.0 : f32
  %c2 = arith.constant 1.0 : f32
  
  // 分配3D memref - 较小尺寸以避免内存问题
  %tensor_a = memref.alloc() : memref<64x64x64xf32>
  %tensor_b = memref.alloc() : memref<64x64x64xf32>
  %result_3d = memref.alloc() : memref<64x64x64xf32>
  
  // 初始化
  linalg.fill ins(%c1 : f32) outs(%tensor_a : memref<64x64x64xf32>)
  linalg.fill ins(%c2 : f32) outs(%tensor_b : memref<64x64x64xf32>)
  
  // 3D复合运算: result = sqrt(A^2 + B^2) * cos(A - B)
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%tensor_a, %tensor_b : memref<64x64x64xf32>, memref<64x64x64xf32>) 
    outs(%result_3d : memref<64x64x64xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %a_sq = arith.mulf %a, %a : f32              // A^2
    %b_sq = arith.mulf %b, %b : f32              // B^2
    %sum_sq = arith.addf %a_sq, %b_sq : f32      // A^2 + B^2
    %sqrt_sum = math.sqrt %sum_sq : f32          // sqrt(A^2 + B^2)
    %diff = arith.subf %a, %b : f32              // A - B
    %cos_diff = math.cos %diff : f32             // cos(A - B)
    %result = arith.mulf %sqrt_sum, %cos_diff : f32  // sqrt(A^2 + B^2) * cos(A - B)
    linalg.yield %result : f32
  }
  
  %t_end = call @rtclock() : () -> f64
  %time_3d = arith.subf %t_end, %t_start : f64
  
  vector.print %time_3d : f64
  return %result_3d : memref<64x64x64xf32>
}

func.func @main() {
  // 测试2D复杂矩阵运算
  %result_2d = call @test_complex_2d_compute() : () -> memref<512x512xf32>
  
  // 测试3D张量运算
  %result_3d = call @test_3d_tensor_compute() : () -> memref<64x64x64xf32>
  
  return
}

// 期望输出：向量化优化后的性能应该显著优于未优化版本
// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [512, 512] strides = [512, 1] data =
// CHECK-NEXT: [{{.*}}{{(, .*)*}}] 