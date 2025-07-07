// ============================================================================
// 外部函数与全局内存声明
// ============================================================================
func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// I=256, J=512
memref.global "private" @A: memref<256xf32> = dense<2.0>
memref.global "private" @B: memref<512xf32> = dense<3.0>
memref.global "private" @C: memref<256x512xf32> = dense<0.0>

// ============================================================================
// 手工优化的外积内核
// 严格遵循 v3.0 分析决策
// ============================================================================
func.func @kernel(%a: memref<256xf32>, %b: memref<512xf32>, %c: memref<256x512xf32>) {
  // --- 计时开始 ---
  %t_start = call @rtclock() : () -> f64
  
  // --- 1. 决策: 并行分块 (Parallel Tiling) ---
  // Tiling Plan: {i:32, j:32}
  affine.parallel (%i0, %j0) = (0, 0) to (256, 512) step (32, 32) {

    // --- 2. 块内迭代 (Intra-Tile Loops) ---
    // 遍历 i-tile
    affine.for %i1 = 0 to 32 {
      %i = arith.addi %i0, %i1 : index
      
      // --- 决策: A 的广播加载 ---
      // 在内层 j 循环开始前加载一次 A(i)，因为它在整个 j-tile 内保持不变
      %a_scalar = memref.load %a[%i] : memref<256xf32>
      %a_vec = vector.splat %a_scalar : vector<8xf32>

      // --- 3. 决策: 向量化计算循环 ---
      // 遍历 j-tile，步长为向量因子 8
      affine.for %j1 = 0 to 32 step 8 {
        %j = arith.addi %j0, %j1 : index
        
        // --- 决策: B 的连续向量加载 ---
        %b_vec = vector.load %b[%j] : memref<512xf32>, vector<8xf32>

        // --- 决策: 向量化计算 ---
        // C(i,j) = A(i) * B(j)
        %result_vec = arith.mulf %a_vec, %b_vec : vector<8xf32>
        
        // --- 决策: C 的连续向量存储 ---
        vector.store %result_vec, %c[%i, %j] : memref<256x512xf32>, vector<8xf32>
      }
    }
  }

  // --- 计时结束 ---
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // --- 打印结果与时间 ---
  %c_cast = memref.cast %c : memref<256x512xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64
  
  return
}

// ============================================================================
// 主入口函数
// ============================================================================
func.func @main() {
  %a_mem = memref.get_global @A : memref<256xf32>
  %b_mem = memref.get_global @B : memref<512xf32>
  %c_mem = memref.get_global @C : memref<256x512xf32>

  call @kernel(%a_mem, %b_mem, %c_mem) : (memref<256xf32>, memref<512xf32>, memref<256x512xf32>) -> ()

  return
}