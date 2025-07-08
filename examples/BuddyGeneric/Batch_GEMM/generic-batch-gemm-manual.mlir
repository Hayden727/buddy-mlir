// ============================================================================
// Map 定义 (用于 affine.min)
// ============================================================================
// Takes d0 (base), s0 (total_size), returns min(d0 + tile_size, total_size)
#map_min_b = affine_map<(d0)[s0] -> (d0 + 4, s0)>
#map_min_m = affine_map<(d0)[s0] -> (d0 + 32, s0)>
#map_min_n = affine_map<(d0)[s0] -> (d0 + 32, s0)>
#map_min_k = affine_map<(d0)[s0] -> (d0 + 64, s0)>

// ============================================================================
// 外部函数与全局内存声明
// ============================================================================
func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

// 定义具体维度 (与 MLIR 版本保持一致)
// Batch=4, M=64, N=128, K=256
memref.global "private" @A: memref<4x64x256xf32> = dense<1.0>
memref.global "private" @B: memref<4x256x128xf32> = dense<2.0>
memref.global "private" @C: memref<4x64x128xf32> = dense<0.0>

// ============================================================================
// 手工优化的 Batch GEMM 内核
// 严格遵循 v3.0 分析决策
// ============================================================================
func.func @kernel(%a: memref<4x64x256xf32>, %b: memref<4x256x128xf32>, %c: memref<4x64x128xf32>) {
  %t_start = call @rtclock() : () -> f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %f0 = arith.constant 0.0 : f32

  // --- 1. 分块循环 (Tiling) ---
  // Tiling Plan: {b:4, m:32, n:32, k:64}
  // 对 B 和 M 进行并行分块，其边界使用 min 表达式处理
  affine.parallel (%b0, %m0) = (0, 0) to (%c4, %c64) step (4, 32) {
    affine.for %n0 = 0 to 128 step 32 {
      affine.for %k0 = 0 to 256 step 64 {
        
        // --- 2. 块内迭代 ---
        // 遍历当前块内的每个元素
        affine.for %b1 = 0 to 4 {
          %b_idx = arith.addi %b0, %b1 : index
          affine.for %m1 = 0 to 32 {
            %m = arith.addi %m0, %m1 : index

            // --- 3. 向量化 N 维度 ---
            // N 维度是向量化维度，步长为 8
            affine.for %n1 = 0 to 32 step 8 {
              %n = arith.addi %n0, %n1 : index

              // --- 4. 掩码处理向量化尾部 ---
              // 计算当前向量操作的实际结束边界 (min(n0 + 32, 128))
              %n_tile_end = affine.min #map_min_n(%n0)[%c128]
              %remaining_n = arith.subi %n_tile_end, %n : index
              %vl = arith.minsi %remaining_n, %c8 : index
              %mask = vector.create_mask %vl : vector<8xi1>

              // --- 5. 准备累加器 (使用 transfer_read 和掩码) ---
              %acc_vec = vector.transfer_read %c[%b_idx, %m, %n], %f0, %mask
                {in_bounds = [false]} // N 维度可能越界
                : memref<4x64x128xf32>, vector<8xf32>

              // --- 6. 归约循环 (K 维度) ---
              // k 维度是归约维度，块内遍历
              %final_vec = affine.for %k1 = 0 to 64 iter_args(%iter_acc = %acc_vec) -> vector<8xf32> {
                %k = arith.addi %k0, %k1 : index

                // --- 7. 加载操作数 ---
                // A: 广播加载 (不依赖 N)
                %a_scalar = memref.load %a[%b_idx, %m, %k] : memref<4x64x256xf32>
                %a_vec = vector.splat %a_scalar : vector<8xf32>

                // B: 连续向量加载 (依赖 N)
                %b_vec = vector.transfer_read %b[%b_idx, %k, %n], %f0, %mask
                  {in_bounds = [false]} // N 维度可能越界
                  : memref<4x256x128xf32>, vector<8xf32>
                
                // FMA 累加
                %next_acc = vector.fma %a_vec, %b_vec, %iter_acc : vector<8xf32>
                affine.yield %next_acc : vector<8xf32>
              }
              
              // --- 8. 存储结果 (使用 transfer_write 和掩码) ---
              vector.transfer_write %final_vec, %c[%b_idx, %m, %n], %mask
                {in_bounds = [false]} // N 维度可能越界
                : vector<8xf32>, memref<4x64x128xf32>
            }
          }
        }
      }
    }
    affine.yield
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  %c_cast = memref.cast %c : memref<4x64x128xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()
  vector.print %time : f64
  return
}

// --- 主入口函数 ---
func.func @main() {
  %a_mem = memref.get_global @A : memref<4x64x256xf32>
  %b_mem = memref.get_global @B : memref<4x256x128xf32>
  %c_mem = memref.get_global @C : memref<4x64x128xf32>

  call @kernel(%a_mem, %b_mem, %c_mem) : (memref<4x64x256xf32>, memref<4x256x128xf32>, memref<4x64x128xf32>) -> ()
  return
}