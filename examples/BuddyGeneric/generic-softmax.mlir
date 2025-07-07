// ============================================================================
// 未优化的 linalg.generic Softmax
// 分解为4个独立的 linalg.generic 操作
// ============================================================================

// --- Trait 定义 (保持不变) ---
#softmax_max_accesses = [ affine_map<(b,f) -> (b,f)>, affine_map<(b,f) -> (b)> ]
#softmax_max_trait = {
  indexing_maps = #softmax_max_accesses,
  iterator_types = ["parallel", "reduction"]
}
#softmax_exp_accesses = [ affine_map<(b,f) -> (b,f)>, affine_map<(b,f) -> (b)>, affine_map<(b,f) -> (b,f)> ]
#softmax_exp_trait = {
  indexing_maps = #softmax_exp_accesses,
  iterator_types = ["parallel", "parallel"]
}
#softmax_sum_accesses = [ affine_map<(b,f) -> (b,f)>, affine_map<(b,f) -> (b)> ]
#softmax_sum_trait = {
  indexing_maps = #softmax_sum_accesses,
  iterator_types = ["parallel", "reduction"]
}
#softmax_div_accesses = [ affine_map<(b,f) -> (b,f)>, affine_map<(b,f) -> (b)>, affine_map<(b,f) -> (b,f)> ]
#softmax_div_trait = {
  indexing_maps = #softmax_div_accesses,
  iterator_types = ["parallel", "parallel"]
}


// --- 外部函数与全局内存 ---
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
// FIX: 移除外部 @expf 的声明

// 定义具体维度: Batch=16, Features=1024
memref.global "private" @INPUT:  memref<16x1024xf32> = dense<1.0>
memref.global "private" @OUTPUT: memref<16x1024xf32> = dense<0.0>


// --- 内核函数 ---
func.func @kernel(%in: memref<16x1024xf32>, %out: memref<16x1024xf32>) {
  %t_start = call @rtclock() : () -> f64

  %max_vals = memref.alloc() : memref<16xf32>
  %exp_vals = memref.alloc() : memref<16x1024xf32>
  %sum_vals = memref.alloc() : memref<16xf32>
  
  %f_min = arith.constant -3.4e38 : f32
  linalg.fill ins(%f_min : f32) outs(%max_vals : memref<16xf32>)

  // --- STAGE 1: 求最大值 ---
  linalg.generic #softmax_max_trait
    ins(%in : memref<16x1024xf32>)
    outs(%max_vals : memref<16xf32>) {
    ^bb0(%in_val: f32, %max_val: f32):
      %is_gt = arith.cmpf ogt, %in_val, %max_val : f32
      %res = arith.select %is_gt, %in_val, %max_val : f32
      linalg.yield %res : f32
  }

  // --- STAGE 2: 减去最大值并求指数 ---
  linalg.generic #softmax_exp_trait
    ins(%in, %max_vals : memref<16x1024xf32>, memref<16xf32>)
    outs(%exp_vals : memref<16x1024xf32>) {
    ^bb0(%in_val: f32, %max_val: f32, %out_unused: f32):
      %sub = arith.subf %in_val, %max_val : f32
      // FIX: 使用 math.exp 操作替代 func.call
      %exp = math.exp %sub : f32
      linalg.yield %exp : f32
  }

  %f_zero = arith.constant 0.0 : f32
  linalg.fill ins(%f_zero : f32) outs(%sum_vals : memref<16xf32>)

  // --- STAGE 3: 求和 ---
  linalg.generic #softmax_sum_trait
    ins(%exp_vals : memref<16x1024xf32>)
    outs(%sum_vals : memref<16xf32>) {
    ^bb0(%in_val: f32, %sum_val: f32):
      %sum = arith.addf %in_val, %sum_val : f32
      linalg.yield %sum : f32
  }

  // --- STAGE 4: 除以和 ---
  linalg.generic #softmax_div_trait
    ins(%exp_vals, %sum_vals : memref<16x1024xf32>, memref<16xf32>)
    outs(%out : memref<16x1024xf32>) {
    ^bb0(%in_val: f32, %sum_val: f32, %out_unused: f32):
      %div = arith.divf %in_val, %sum_val : f32
      linalg.yield %div : f32
  }

  memref.dealloc %max_vals : memref<16xf32>
  memref.dealloc %exp_vals : memref<16x1024xf32>
  memref.dealloc %sum_vals : memref<16xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  %out_cast = memref.cast %out : memref<16x1024xf32> to memref<*xf32>
  call @printMemrefF32(%out_cast) : (memref<*xf32>) -> ()
  vector.print %time : f64

  return
}

// --- 主入口函数 ---
func.func @main() {
  %in_mem = memref.get_global @INPUT : memref<16x1024xf32>
  %out_mem = memref.get_global @OUTPUT : memref<16x1024xf32>
  call @kernel(%in_mem, %out_mem) : (memref<16x1024xf32>, memref<16x1024xf32>) -> ()
  return
}