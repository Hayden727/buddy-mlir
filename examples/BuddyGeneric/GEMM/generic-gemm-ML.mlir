// ============================================================================
// GEMM - 大规模 (Large) - M=256, N=512, K=1024
// 包含未优化和优化版本
// ============================================================================

// 外部函数声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

#gemm_accesses = [
  affine_map<(d0, d1, d2) -> (d0, d2)>,
  affine_map<(d0, d1, d2) -> (d2, d1)>,
  affine_map<(d0, d1, d2) -> (d0, d1)>
]
#gemm_trait = {
  indexing_maps = #gemm_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

memref.global "private" @A_large_unopt: memref<256x1024xf32> = dense<2.0>
memref.global "private" @B_large_unopt: memref<1024x512xf32> = dense<3.0>
memref.global "private" @C_large_unopt: memref<256x512xf32> = dense<1.0>

func.func @kernel_large_unopt(%a: memref<256x1024xf32>, %b: memref<1024x512xf32>, %c: memref<256x512xf32>) {
  %t_start = call @rtclock() : () -> f64
  linalg.generic #gemm_trait
    ins(%a, %b : memref<256x1024xf32>, memref<1024x512xf32>)
    outs(%c : memref<256x512xf32>) {
      ^bb0(%in_a: f32, %in_b: f32, %out_c: f32):
        %p = arith.mulf %in_a, %in_b : f32
        %s = arith.addf %out_c, %p : f32
        linalg.yield %s : f32
  }
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  %c_cast = memref.cast %c : memref<256x512xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()
  vector.print %time : f64
  return
}

func.func @main() { // 未优化版本的入口
  %a = memref.get_global @A_large_unopt : memref<256x1024xf32>
  %b = memref.get_global @B_large_unopt : memref<1024x512xf32>
  %c = memref.get_global @C_large_unopt : memref<256x512xf32>
  call @kernel_large_unopt(%a, %b, %c) : (memref<256x1024xf32>, memref<1024x512xf32>, memref<256x512xf32>) -> ()
  return
}