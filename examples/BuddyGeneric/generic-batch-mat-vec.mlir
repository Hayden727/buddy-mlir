// ============================================================================
// 未优化的 linalg.generic 批量矩阵-向量乘
// C(b, i) += A(b, i, k) * B(b, k)
// ============================================================================

// 1. Linalg Generic 操作的 "trait" (特性) 定义
// 维度映射: (d0,d1,d2) -> (b,i,k)
#batch_mv_accesses = [
  // A: (b, i, k) -> (b, i, k)
  affine_map<(d0,d1,d2) -> (d0,d1,d2)>,
  // B: (b, i, k) -> (b, k)
  affine_map<(d0,d1,d2) -> (d0,d2)>,
  // C: (b, i, k) -> (b, i)
  affine_map<(d0,d1,d2) -> (d0,d1)>
]

#batch_mv_trait = {
  indexing_maps = #batch_mv_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// 2. 外部函数与全局内存声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

// 定义具体维度
// Batch=16, I=128, K=256
memref.global "private" @A: memref<16x128x256xf32> = dense<1.0>
memref.global "private" @B: memref<16x256xf32> = dense<2.0>
memref.global "private" @C: memref<16x128xf32> = dense<0.0>

// 3. 执行计算并计时的内核函数
func.func @kernel(%a: memref<16x128x256xf32>, %b: memref<16x256xf32>, %c: memref<16x128xf32>) {
  %t_start = call @rtclock() : () -> f64

  linalg.generic #batch_mv_trait
    ins(%a, %b : memref<16x128x256xf32>, memref<16x256xf32>)
    outs(%c : memref<16x128xf32>) {
      ^bb0(%in_a: f32, %in_b: f32, %out_c: f32):
        %product = arith.mulf %in_a, %in_b : f32
        %sum = arith.addf %out_c, %product : f32
        linalg.yield %sum : f32
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %c_cast = memref.cast %c : memref<16x128xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64

  return
}

// 4. 主入口函数
func.func @main() {
  %a_mem = memref.get_global @A : memref<16x128x256xf32>
  %b_mem = memref.get_global @B : memref<16x256xf32>
  %c_mem = memref.get_global @C : memref<16x128xf32>

  call @kernel(%a_mem, %b_mem, %c_mem) : (memref<16x128x256xf32>, memref<16x256xf32>, memref<16x128xf32>) -> ()

  return
}