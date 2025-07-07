// ============================================================================
// 外部函数与全局内存声明
// ============================================================================
func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// N=1, OC=64, H=28, W=28
// IC=32, KH=3, KW=3
// Input H/W = 28+3-1 = 30
memref.global "private" @INPUT:  memref<1x32x30x30xf32> = dense<1.0>
memref.global "private" @FILTER: memref<64x32x3x3xf32> = dense<2.0>
memref.global "private" @OUTPUT: memref<1x64x28x28xf32> = dense<0.0>

// ============================================================================
// 手工优化的 Conv2D 内核
// 严格遵循 v3.0 分析决策
// ============================================================================
func.func @kernel(%in: memref<1x32x30x30xf32>, %fil: memref<64x32x3x3xf32>, %out: memref<1x64x28x28xf32>) {
  // --- 计时开始 ---
  %t_start = call @rtclock() : () -> f64

  // --- 常量定义 ---
  %c0 = arith.constant 0 : index
  
  // --- 并行分块 (Parallel Tiling) ---
  // Tiling Plan: {oc:16, h:7}
  affine.parallel (%oc0, %h0) = (0, 0) to (64, 28) step (16, 7) {

    // w 维度分块
    // Tiling Plan: {w:16}
    affine.for %w0 = 0 to 28 step 16 {

      // ic 维度分块
      // Tiling Plan: {ic:8}
      affine.for %ic0 = 0 to 32 step 8 {

        // --- 块内迭代 (Intra-Tile Loops) ---
        affine.for %oc1 = 0 to 16 {
          %oc = arith.addi %oc0, %oc1 : index
          affine.for %h1 = 0 to 7 {
            %h = arith.addi %h0, %h1 : index
            
            // 向量化 w 维度, 步长为 8
            affine.for %w1 = 0 to 16 step 8 {
              %w = arith.addi %w0, %w1 : index

              %out_vec = vector.load %out[%c0, %oc, %h, %w] : memref<1x64x28x28xf32>, vector<8xf32>

              // --- 归约循环 ---
              %final_vec = affine.for %ic1 = 0 to 8 iter_args(%acc0 = %out_vec) -> vector<8xf32> {
                %ic = arith.addi %ic0, %ic1 : index
                %acc1 = affine.for %kh = 0 to 3 iter_args(%tmp_acc1 = %acc0) -> vector<8xf32> {
                  %acc2 = affine.for %kw = 0 to 3 iter_args(%tmp_acc2 = %tmp_acc1) -> vector<8xf32> {
                    
                    %ih = arith.addi %h, %kh : index
                    %iw = arith.addi %w, %kw : index
                    %in_vec = vector.load %in[%c0, %ic, %ih, %iw] : memref<1x32x30x30xf32>, vector<8xf32>

                    %fil_scalar = memref.load %fil[%oc, %ic, %kh, %kw] : memref<64x32x3x3xf32>
                    %fil_vec = vector.splat %fil_scalar : vector<8xf32>
                    
                    %next_acc = vector.fma %in_vec, %fil_vec, %tmp_acc2 : vector<8xf32>
                    affine.yield %next_acc : vector<8xf32>
                  }
                  affine.yield %acc2 : vector<8xf32>
                }
                affine.yield %acc1 : vector<8xf32>
              }

              vector.store %final_vec, %out[%c0, %oc, %h, %w] : memref<1x64x28x28xf32>, vector<8xf32>
            }
          }
        }
      }
    }
  }

  // --- 计时结束 ---
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // --- 打印计算结果 ---
  // FIX: 将打印结果的逻辑移到此处
  %out_cast = memref.cast %out : memref<1x64x28x28xf32> to memref<*xf32>
  call @printMemrefF32(%out_cast) : (memref<*xf32>) -> ()

  // --- 打印时间 ---
  vector.print %time : f64
  
  return
}

// ============================================================================
// 主入口函数
// ============================================================================
func.func @main() {
  %in_mem = memref.get_global @INPUT : memref<1x32x30x30xf32>
  %fil_mem = memref.get_global @FILTER : memref<64x32x3x3xf32>
  %out_mem = memref.get_global @OUTPUT : memref<1x64x28x28xf32>

  call @kernel(%in_mem, %fil_mem, %out_mem) : (memref<1x32x30x30xf32>, memref<64x32x3x3xf32>, memref<1x64x28x28xf32>) -> ()

  return
}