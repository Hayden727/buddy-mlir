// ============================================================================
// 未优化的 linalg.generic 外积
// C(i, j) = A(i) * B(j)
// ============================================================================

// 1. Linalg Generic 操作的 "trait" (特性) 定义
// 维度映射: (d0,d1) -> (i,j)
#outer_product_accesses = [
  // A: (i, j) -> (i)
  affine_map<(d0,d1) -> (d0)>,
  // B: (i, j) -> (j)
  affine_map<(d0,d1) -> (d1)>,
  // C: (i, j) -> (i, j)
  affine_map<(d0,d1) -> (d0,d1)>
]

#outer_product_trait = {
  indexing_maps = #outer_product_accesses,
  iterator_types = ["parallel", "parallel"]
}

// 2. 外部函数与全局内存声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

// 定义具体维度
// I=256, J=512
memref.global "private" @A: memref<256xf32> = dense<2.0>
memref.global "private" @B: memref<512xf32> = dense<3.0>
memref.global "private" @C: memref<256x512xf32> = dense<0.0>

// 3. 执行计算并计时的内核函数
func.func @kernel(%a: memref<256xf32>, %b: memref<512xf32>, %c: memref<256x512xf32>) {
  %t_start = call @rtclock() : () -> f64

  linalg.generic #outer_product_trait
    ins(%a, %b : memref<256xf32>, memref<512xf32>)
    outs(%c : memref<256x512xf32>) {
      ^bb0(%in_a: f32, %in_b: f32, %out_c_unused: f32): // %out_c_unused 被忽略
        %prod = arith.mulf %in_a, %in_b : f32
        linalg.yield %prod : f32 // 直接 yield 乘积
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %c_cast = memref.cast %c : memref<256x512xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64

  return
}

// 4. 主入口函数
func.func @main() {
  %a_mem = memref.get_global @A : memref<256xf32>
  %b_mem = memref.get_global @B : memref<512xf32>
  %c_mem = memref.get_global @C : memref<256x512xf32>

  call @kernel(%a_mem, %b_mem, %c_mem) : (memref<256xf32>, memref<512xf32>, memref<256x512xf32>) -> ()

  return
}