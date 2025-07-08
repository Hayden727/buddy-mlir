// ============================================================================
// 外部函数与全局内存声明
// ============================================================================
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

memref.global "private" @A: memref<4096xf32> = dense<2.0>
memref.global "private" @B: memref<4096xf32> = dense<3.0>
memref.global "private" @C: memref<f32> = dense<0.0>

// ============================================================================
// 手工优化的点积内核
// 严格遵循 v3.0 分析决策
// ============================================================================
func.func @kernel(%a: memref<4096xf32>, %b: memref<4096xf32>, %c: memref<f32>) {
  // --- 计时开始 ---
  %t_start = call @rtclock() : () -> f64

  // --- 决策: C 的处理 ---
  // 1. 将 C 的初始值加载到标量累加器
  %c_init_scalar = memref.load %c[] : memref<f32>
  // 2. 将标量累加器广播成一个向量累加器。所有分步计算的结果将累加到这里。
  %acc_vec_init = vector.splat %c_init_scalar : vector<8xf32>

  // --- 决策: 向量化归约循环 (无分块) ---
  // 使用带 iter_args 的 affine.for 来维护向量累加器的状态
  %final_acc_vec = affine.for %i = 0 to 4096 step 8 iter_args(%acc_vec = %acc_vec_init) -> vector<8xf32> {
    
    // --- 决策: 向量加载 ---
    %a_vec = vector.load %a[%i] : memref<4096xf32>, vector<8xf32>
    %b_vec = vector.load %b[%i] : memref<4096xf32>, vector<8xf32>
    
    // --- 决策: 向量化 FMA 累加 ---
    // acc = a*b + acc
    %next_acc_vec = vector.fma %a_vec, %b_vec, %acc_vec : vector<8xf32>
    
    affine.yield %next_acc_vec : vector<8xf32>
  }
  
  // --- 决策: 水平归约 ---
  // 将最终向量累加器 %final_acc_vec 的所有元素相加，得到一个标量结果
  %final_scalar = vector.reduction <add>, %final_acc_vec : vector<8xf32> into f32
  
  // --- 决策: 存储最终结果 ---
  // 将归约后的标量结果存回 C
  memref.store %final_scalar, %c[] : memref<f32>
  
  // --- 计时结束 ---
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // --- 打印结果与时间 ---
  %c_cast = memref.cast %c : memref<f32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64
  
  return
}

// ============================================================================
// 主入口函数
// ============================================================================
func.func @main() {
  %a_mem = memref.get_global @A : memref<4096xf32>
  %b_mem = memref.get_global @B : memref<4096xf32>
  %c_mem = memref.get_global @C : memref<f32>

  call @kernel(%a_mem, %b_mem, %c_mem) : (memref<4096xf32>, memref<4096xf32>, memref<f32>) -> ()

  return
}