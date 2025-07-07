// ============================================================================
// 外部函数与全局内存声明
// ============================================================================
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

// 定义具体维度: Batch=16, Features=1024
memref.global "private" @INPUT:  memref<16x1024xf32> = dense<1.0>
memref.global "private" @OUTPUT: memref<16x1024xf32> = dense<0.0>

// ============================================================================
// 手工优化的 Softmax 内核
// 严格遵循 v3.0 分析决策
// ============================================================================
func.func @kernel(%in: memref<16x1024xf32>, %out: memref<16x1024xf32>) {
  // --- 计时开始 ---
  %t_start = call @rtclock() : () -> f64

  // --- 常量定义 ---
  %cst_fmin = arith.constant -3.4e38 : f32
  %cst_fzero = arith.constant 0.0 : f32
  %vec_fmin = vector.splat %cst_fmin : vector<8xf32>
  %vec_fzero = vector.splat %cst_fzero : vector<8xf32>

  // --- 并行化 Batch 维度 ---
  affine.parallel (%b) = (0) to (16) {

    // ================= STAGE 1: 求最大值 =================
    %max_vec_acc = affine.for %f = 0 to 1024 step 8 iter_args(%acc = %vec_fmin) -> vector<8xf32> {
      %in_vec = vector.load %in[%b, %f] : memref<16x1024xf32>, vector<8xf32>
      // FIX: 使用正确的 arith.maximumf 操作
      %next_acc = arith.maximumf %acc, %in_vec : vector<8xf32>
      affine.yield %next_acc : vector<8xf32>
    }
    %max_scalar = vector.reduction <maximumf>, %max_vec_acc : vector<8xf32> into f32

    // ================= STAGE 2: 减去最大值并求指数 =================
    %max_val_vec = vector.splat %max_scalar : vector<8xf32>
    affine.for %f = 0 to 1024 step 8 {
      %in_vec = vector.load %in[%b, %f] : memref<16x1024xf32>, vector<8xf32>
      %sub_vec = arith.subf %in_vec, %max_val_vec : vector<8xf32>
      %exp_vec = math.exp %sub_vec : vector<8xf32>
      vector.store %exp_vec, %out[%b, %f] : memref<16x1024xf32>, vector<8xf32>
    }

    // ================= STAGE 3: 求和 =================
    %sum_vec_acc = affine.for %f = 0 to 1024 step 8 iter_args(%acc = %vec_fzero) -> vector<8xf32> {
      %exp_vec = vector.load %out[%b, %f] : memref<16x1024xf32>, vector<8xf32>
      %next_acc = arith.addf %acc, %exp_vec : vector<8xf32>
      affine.yield %next_acc : vector<8xf32>
    }
    %sum_scalar = vector.reduction <add>, %sum_vec_acc : vector<8xf32> into f32

    // ================= STAGE 4: 除以和 =================
    %sum_val_vec = vector.splat %sum_scalar : vector<8xf32>
    affine.for %f = 0 to 1024 step 8 {
      %exp_vec = vector.load %out[%b, %f] : memref<16x1024xf32>, vector<8xf32>
      %div_vec = arith.divf %exp_vec, %sum_val_vec : vector<8xf32>
      vector.store %div_vec, %out[%b, %f] : memref<16x1024xf32>, vector<8xf32>
    }
    
  }

  // --- 计时结束 ---
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // --- 打印结果与时间 ---
  %out_cast = memref.cast %out : memref<16x1024xf32> to memref<*xf32>
  call @printMemrefF32(%out_cast) : (memref<*xf32>) -> ()
  vector.print %time : f64
  
  return
}

// ============================================================================
// 主入口函数
// ============================================================================
func.func @main() {
  %in_mem = memref.get_global @INPUT : memref<16x1024xf32>
  %out_mem = memref.get_global @OUTPUT : memref<16x1024xf32>
  call @kernel(%in_mem, %out_mem) : (memref<16x1024xf32>, memref<16x1024xf32>) -> ()
  return
}