// ============================================================================
// 未优化的 linalg.generic 点积
// c += A(i) * B(i)
// ============================================================================

// 1. Linalg Generic 操作的 "trait" (特性) 定义
// 维度映射: (d0) -> (i)
#dot_accesses = [
  // A: (i) -> (i)
  affine_map<(d0) -> (d0)>,
  // B: (i) -> (i)
  affine_map<(d0) -> (d0)>,
  // C (output): () -> ()  (0-D, 不依赖循环变量)
  affine_map<(d0) -> ()>
]

#dot_trait = {
  indexing_maps = #dot_accesses,
  iterator_types = ["reduction"]
}

// 2. 外部函数与全局内存声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

// 定义具体维度
memref.global "private" @A: memref<4096xf32> = dense<2.0>
memref.global "private" @B: memref<4096xf32> = dense<3.0>
// 输出是一个标量，用含单个元素的memref表示，并初始化为0
memref.global "private" @C: memref<f32> = dense<0.0>

// 3. 执行点积计算并计时的内核函数
func.func @kernel(%a_mem: memref<4096xf32>, %b_mem: memref<4096xf32>, %c_mem: memref<f32>) {
  %t_start = call @rtclock() : () -> f64

  linalg.generic #dot_trait
    ins(%a_mem, %b_mem : memref<4096xf32>, memref<4096xf32>)
    outs(%c_mem : memref<f32>) {
      ^bb0(%a_in: f32, %b_in: f32, %c_in: f32):
        %prod = arith.mulf %a_in, %b_in : f32
        %sum = arith.addf %c_in, %prod : f32
        linalg.yield %sum : f32
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // 为了能被printMemrefF32接受，需要将0-D memref转换为1-D memref
  // 这在实际中可能需要更复杂的处理，这里我们假设打印函数能处理
  %c_cast = memref.cast %c_mem : memref<f32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64

  return
}

// 4. 主入口函数
func.func @main() {
  %a = memref.get_global @A : memref<4096xf32>
  %b = memref.get_global @B : memref<4096xf32>
  %c = memref.get_global @C : memref<f32>

  call @kernel(%a, %b, %c) : (memref<4096xf32>, memref<4096xf32>, memref<f32>) -> ()

  return
}