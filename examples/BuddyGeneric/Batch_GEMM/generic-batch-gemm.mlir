// ============================================================================
// 未优化的 linalg.generic 批量矩阵-矩阵乘法 (Batch GEMM)
// C(b, m, n) += A(b, m, k) * B(b, k, n)
// ============================================================================

// 1. Linalg Generic 操作的 "trait" (特性) 定义
// 维度映射: (d0,d1,d2,d3) -> (b,m,n,k)
#batch_gemm_accesses = [
  // A: (b, m, k) -> (d0, d1, d3)
  affine_map<(d0,d1,d2,d3) -> (d0, d1, d3)>,
  // B: (b, k, n) -> (d0, d3, d2)
  affine_map<(d0,d1,d2,d3) -> (d0, d3, d2)>,
  // C: (b, m, n) -> (d0, d1, d2)
  affine_map<(d0,d1,d2,d3) -> (d0, d1, d2)>
]

#batch_gemm_trait = {
  indexing_maps = #batch_gemm_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
}

// 2. 外部函数与全局内存声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

// 定义具体维度
// 示例尺寸: Batch=4, M=64, N=128, K=256
memref.global "private" @A: memref<4x64x256xf32> = dense<1.0>
memref.global "private" @B: memref<4x256x128xf32> = dense<2.0>
memref.global "private" @C: memref<4x64x128xf32> = dense<0.0>

// 3. 执行 Batch GEMM 计算并计时的内核函数
func.func @kernel(%a: memref<4x64x256xf32>, %b: memref<4x256x128xf32>, %c: memref<4x64x128xf32>) {
  %t_start = call @rtclock() : () -> f64

  linalg.generic #batch_gemm_trait
    ins(%a, %b : memref<4x64x256xf32>, memref<4x256x128xf32>)
    outs(%c : memref<4x64x128xf32>) {
      ^bb0(%in_a: f32, %in_b: f32, %out_c: f32):
        %product = arith.mulf %in_a, %in_b : f32
        %sum = arith.addf %out_c, %product : f32
        linalg.yield %sum : f32
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %c_cast = memref.cast %c : memref<4x64x128xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64

  return
}

// 4. 主入口函数
func.func @main() {
  %a_mem = memref.get_global @A : memref<4x64x256xf32>
  %b_mem = memref.get_global @B : memref<4x256x128xf32>
  %c_mem = memref.get_global @C : memref<4x64x128xf32>

  call @kernel(%a_mem, %b_mem, %c_mem) : (memref<4x64x256xf32>, memref<4x256x128xf32>, memref<4x64x128xf32>) -> ()

  return
}