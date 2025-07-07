// ============================================================================
// 外部函数与全局内存声明
// ============================================================================
func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

memref.global "private" @X: memref<4096xf32> = dense<3.0>
memref.global "private" @Y: memref<4096xf32> = dense<10.0>

// ============================================================================
// 手工优化的 AXPY 内核
// 严格遵循 v3.0 分析决策
// ============================================================================
func.func @kernel(%a: f32, %x: memref<4096xf32>, %y: memref<4096xf32>) {
  // --- 计时开始 ---
  %t_start = call @rtclock() : () -> f64

  // --- 决策: 标量 a 的处理 ---
  // 将外部标量 a 广播成一个向量，以便在循环内使用
  %a_vec = vector.splat %a : vector<8xf32>

  // --- 决策: 向量化循环 (无分块) ---
  // 唯一的循环 i 被向量化，步长为向量因子 8
  affine.for %i = 0 to 4096 step 8 {

    // --- 决策: 向量加载 ---
    // X 和 Y 都是连续向量加载
    %x_vec = vector.load %x[%i] : memref<4096xf32>, vector<8xf32>
    %y_vec = vector.load %y[%i] : memref<4096xf32>, vector<8xf32>
    
    // --- 决策: 向量化计算 ---
    // 使用 FMA 指令实现 y = a*x + y
    %result_vec = vector.fma %a_vec, %x_vec, %y_vec : vector<8xf32>
    
    // --- 决策: 向量存储 ---
    // 将结果向量写回 Y
    vector.store %result_vec, %y[%i] : memref<4096xf32>, vector<8xf32>
  }
  
  // --- 计时结束 ---
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // --- 打印结果与时间 ---
  %y_cast = memref.cast %y : memref<4096xf32> to memref<*xf32>
  call @printMemrefF32(%y_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64
  
  return
}

// ============================================================================
// 主入口函数
// ============================================================================
func.func @main() {
  %a_scalar = arith.constant 2.0 : f32
  %x_mem = memref.get_global @X : memref<4096xf32>
  %y_mem = memref.get_global @Y : memref<4096xf32>

  call @kernel(%a_scalar, %x_mem, %y_mem) : (f32, memref<4096xf32>, memref<4096xf32>) -> ()

  return
}