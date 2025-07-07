// ============================================================================
// 外部函数与全局内存声明
// ============================================================================
func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// Batch=16, I=128, K=256
memref.global "private" @A: memref<16x128x256xf32> = dense<1.0>
memref.global "private" @B: memref<16x256xf32> = dense<2.0>
memref.global "private" @C: memref<16x128xf32> = dense<0.0>

// ============================================================================
// 手工优化的批量矩阵-向量乘内核
// 严格遵循 v3.0 分析决策
// ============================================================================
func.func @kernel(%a: memref<16x128x256xf32>, %b: memref<16x256xf32>, %c: memref<16x128xf32>) {
  // --- 计时开始 ---
  %t_start = call @rtclock() : () -> f64

  %cst_0_f32 = arith.constant 0.0 : f32
  
  // --- 并行分块 (Parallel Tiling) ---
  // Tiling Plan: {b:4, i:32}
  affine.parallel (%b0, %i0) = (0, 0) to (16, 128) step (4, 32) {

    // k 维度分块
    // Tiling Plan: {k:32}
    affine.for %k0 = 0 to 256 step 32 {

      // 块内迭代
      affine.for %b1 = 0 to 4 {
        %batch_idx = arith.addi %b0, %b1 : index

        // 向量化计算循环
        affine.for %i1 = 0 to 32 step 8 {
          %i_base = arith.addi %i0, %i1 : index

          %c_vec = vector.load %c[%batch_idx, %i_base] : memref<16x128xf32>, vector<8xf32>

          // k-tile 归约循环
          %final_vec = affine.for %k1 = 0 to 32 iter_args(%acc_vec = %c_vec) -> vector<8xf32> {
            %k_idx = arith.addi %k0, %k1 : index

            // A 的跨步向量加载 (Strided Vector Load)
            %a_vec_init = vector.splat %cst_0_f32 : vector<8xf32>
            %a_vec = affine.for %v = 0 to 8 iter_args(%tmp_vec = %a_vec_init) -> vector<8xf32> {
              %i_offset = arith.addi %i_base, %v : index
              %a_scalar = memref.load %a[%batch_idx, %i_offset, %k_idx] : memref<16x128x256xf32>
              // FIX: 使用方括号语法处理动态索引
              %next_vec = vector.insertelement %a_scalar, %tmp_vec[%v : index] : vector<8xf32>
              affine.yield %next_vec : vector<8xf32>
            }

            // B 的广播加载
            %b_scalar = memref.load %b[%batch_idx, %k_idx] : memref<16x256xf32>
            %b_vec = vector.splat %b_scalar : vector<8xf32>

            // FMA 累加
            %next_acc = vector.fma %a_vec, %b_vec, %acc_vec : vector<8xf32>
            affine.yield %next_acc : vector<8xf32>
          }

          // 存储结果
          vector.store %final_vec, %c[%batch_idx, %i_base] : memref<16x128xf32>, vector<8xf32>
        }
      }
    }
  }

  // --- 计时结束 ---
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // --- 打印结果与时间 ---
  %c_cast = memref.cast %c : memref<16x128xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()

  vector.print %time : f64
  
  return
}

// ============================================================================
// 主入口函数
// ============================================================================
func.func @main() {
  %a_mem = memref.get_global @A : memref<16x128x256xf32>
  %b_mem = memref.get_global @B : memref<16x256xf32>
  %c_mem = memref.get_global @C : memref<16x128xf32>

  call @kernel(%a_mem, %b_mem, %c_mem) : (memref<16x128x256xf32>, memref<16x256xf32>, memref<16x128xf32>) -> ()

  return
}