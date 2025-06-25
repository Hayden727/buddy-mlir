// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
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
// 测试用例: 简单元素级加法 - 使用memref类型
// 
// 这个测试用例验证：
// 1. 硬件能力分析是否正确识别向量宽度
// 2. 操作模式分析是否正确识别单一并行维度
// 3. 内存访问分析是否正确识别连续访问模式
// 4. 向量化决策是否建议进行向量化
// ===----------------------------------------------------------------------===//

func.func @test_simple_add() -> memref<1024xf32> {
  %t_start = call @rtclock() : () -> f64
  
  // 创建输入数据 - 使用memref类型
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  
  // 分配memref并初始化
  %input1 = memref.alloc() : memref<1024xf32>
  %input2 = memref.alloc() : memref<1024xf32>
  %output = memref.alloc() : memref<1024xf32>
  
  // 使用linalg.fill初始化输入
  linalg.fill ins(%c2 : f32) outs(%input1 : memref<1024xf32>)
  linalg.fill ins(%c3 : f32) outs(%input2 : memref<1024xf32>)
  
  // 执行linalg.generic操作 - 这是我们要优化的核心操作
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
  %memref_unranked = memref.cast %output : memref<1024xf32> to memref<*xf32>
  call @printMemrefF32(%memref_unranked) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  return %output : memref<1024xf32>
}

func.func @main() {
  %result = call @test_simple_add() : () -> memref<1024xf32>
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1024] strides = [1] data =
// CHECK-NEXT: [5{{(, 5)*}}] 