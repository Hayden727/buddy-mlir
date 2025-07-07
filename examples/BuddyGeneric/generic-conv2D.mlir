// ============================================================================
// 未优化的 linalg.generic Conv2D
// C(n,oc,h,w) += A(n,ic,h+kh,w+kw) * B(oc,ic,kh,kw)
// ============================================================================

// 1. Linalg Generic 操作的 "trait" (特性) 定义
// 维度映射: (d0,d1,d2,d3,d4,d5,d6) -> (n,oc,h,w,ic,kh,kw)
#conv_accesses = [
  // Input: (n, ic, h+kh, w+kw)
  affine_map<(d0,d1,d2,d3,d4,d5,d6) -> (d0, d4, d2+d5, d3+d6)>,
  // Filter: (oc, ic, kh, kw)
  affine_map<(d0,d1,d2,d3,d4,d5,d6) -> (d1, d4, d5, d6)>,
  // Output: (n, oc, h, w)
  affine_map<(d0,d1,d2,d3,d4,d5,d6) -> (d0, d1, d2, d3)>
]

#conv_trait = {
  indexing_maps = #conv_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
}

// 2. 外部函数与全局内存声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

// 定义具体维度
// N=1, OC=64, H=28, W=28
// IC=32, KH=3, KW=3
// Input H/W = Output H/W + Kernel H/W - 1 = 28+3-1 = 30
memref.global "private" @INPUT:  memref<1x32x30x30xf32> = dense<1.0>
memref.global "private" @FILTER: memref<64x32x3x3xf32> = dense<2.0>
memref.global "private" @OUTPUT: memref<1x64x28x28xf32> = dense<0.0>

// 3. 执行 Conv2D 计算并计时的内核函数
func.func @kernel(%in: memref<1x32x30x30xf32>, %fil: memref<64x32x3x3xf32>, %out: memref<1x64x28x28xf32>) {
  %t_start = call @rtclock() : () -> f64

  // linalg.generic 对 memref 进行原地更新
  linalg.generic #conv_trait
    ins(%in, %fil : memref<1x32x30x30xf32>, memref<64x32x3x3xf32>)
    outs(%out : memref<1x64x28x28xf32>) {
      // 区域内的计算逻辑与 GEMM 相同
      ^bb0(%in_val: f32, %fil_val: f32, %out_val: f32):
        %product = arith.mulf %in_val, %fil_val : f32
        %sum = arith.addf %out_val, %product : f32
        linalg.yield %sum : f32
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %out_cast = memref.cast %out : memref<1x64x28x28xf32> to memref<*xf32>
  call @printMemrefF32(%out_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64

  return
}

// 4. 主入口函数
func.func @main() {
  %in_mem = memref.get_global @INPUT : memref<1x32x30x30xf32>
  %fil_mem = memref.get_global @FILTER : memref<64x32x3x3xf32>
  %out_mem = memref.get_global @OUTPUT : memref<1x64x28x28xf32>

  call @kernel(%in_mem, %fil_mem, %out_mem) : (memref<1x32x30x30xf32>, memref<64x32x3x3xf32>, memref<1x64x28x28xf32>) -> ()

  return
}