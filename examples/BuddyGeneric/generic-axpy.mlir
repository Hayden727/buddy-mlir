// ============================================================================
// 未优化的 linalg.generic AXPY
// Y(i) = a * X(i) + Y(i)
// ============================================================================

// 1. Linalg Generic 操作的 "trait" (特性) 定义
// 维度映射: (d0) -> (i)
#axpy_accesses = [
  // X: (i) -> (i)
  affine_map<(d0) -> (d0)>,
  // Y: (i) -> (i)
  affine_map<(d0) -> (d0)>
]

#axpy_trait = {
  indexing_maps = #axpy_accesses,
  iterator_types = ["parallel"]
}

// 2. 外部函数与全局内存声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

// 定义具体维度
memref.global "private" @X: memref<4096xf32> = dense<3.0>
memref.global "private" @Y: memref<4096xf32> = dense<10.0>

// 3. 执行 AXPY 计算并计时的内核函数
func.func @kernel(%a: f32, %x_mem: memref<4096xf32>, %y_mem: memref<4096xf32>) {
  %t_start = call @rtclock() : () -> f64

  // ins/outs 只包含memref。标量 %a 被区域隐式捕获。
  linalg.generic #axpy_trait
    ins(%x_mem : memref<4096xf32>)
    outs(%y_mem : memref<4096xf32>) {
      ^bb0(%x_in: f32, %y_in: f32):  // %x_in来自%x_mem, %y_in来自%y_mem
        // 区域内的计算逻辑
        %prod = arith.mulf %a, %x_in : f32       // %a 是从外部作用域捕获的
        %sum = arith.addf %y_in, %prod : f32
        linalg.yield %sum : f32
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %y_cast = memref.cast %y_mem : memref<4096xf32> to memref<*xf32>
  call @printMemrefF32(%y_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64

  return
}

// 4. 主入口函数
func.func @main() {
  // 定义标量 a
  %a_scalar = arith.constant 2.0 : f32

  %x = memref.get_global @X : memref<4096xf32>
  %y = memref.get_global @Y : memref<4096xf32>

  call @kernel(%a_scalar, %x, %y) : (f32, memref<4096xf32>, memref<4096xf32>) -> ()

  return
}