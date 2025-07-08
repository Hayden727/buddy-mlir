// ============================================================================
// GEMM - 极小规模 (Very Small) - 未优化
// M=32, N=64, K=128
// ============================================================================

// 定义 GEMM (C += A * B) 的 linalg 结构特性
#gemm_accesses = [
  affine_map<(d0, d1, d2) -> (d0, d2)>,   // A(m, k)
  affine_map<(d0, d1, d2) -> (d2, d1)>,   // B(k, n)
  affine_map<(d0, d1, d2) -> (d0, d1)>    // C(m, n)
]

#gemm_trait = {
  indexing_maps = #gemm_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// 外部函数声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

// 全局内存区域
memref.global "private" @A_vsmall: memref<32x128xf32> = dense<2.0>
memref.global "private" @B_vsmall: memref<128x64xf32> = dense<3.0>
memref.global "private" @C_vsmall: memref<32x64xf32> = dense<1.0>

// 执行 GEMM 计算并计时的内核函数
func.func @kernel_vsmall(%a: memref<32x128xf32>, %b: memref<128x64xf32>, %c: memref<32x64xf32>) {
  %t_start = call @rtclock() : () -> f64

  linalg.generic #gemm_trait
    ins(%a, %b : memref<32x128xf32>, memref<128x64xf32>)
    outs(%c : memref<32x64xf32>) {
      ^bb0(%in_a: f32, %in_b: f32, %out_c: f32):
        %product = arith.mulf %in_a, %in_b : f32
        %sum = arith.addf %out_c, %product : f32
        linalg.yield %sum : f32
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %c_cast = memref.cast %c : memref<32x64xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64

  return
}

// 主入口函数
func.func @main() {
  %a_mem = memref.get_global @A_vsmall : memref<32x128xf32>
  %b_mem = memref.get_global @B_vsmall : memref<128x64xf32>
  %c_mem = memref.get_global @C_vsmall : memref<32x64xf32>

  call @kernel_vsmall(%a_mem, %b_mem, %c_mem) : (memref<32x128xf32>, memref<128x64xf32>, memref<32x64xf32>) -> ()

  return
}