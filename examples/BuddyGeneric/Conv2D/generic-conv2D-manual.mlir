// ============================================================================
// Map 定义
// ============================================================================
#map_w_bound = affine_map<(d0) -> (d0 + 16, 28)>
#map_ic_bound = affine_map<(d0) -> (d0 + 8)>
#map_id = affine_map<(d0) -> (d0)>

// ============================================================================
// 外部函数与全局内存声明
// ============================================================================
func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// N=1, OC=64, H=28, W=28, IC=32, KH=3, KW=3, Input H/W=30
memref.global "private" @INPUT:  memref<1x32x30x30xf32> = dense<1.0>
memref.global "private" @FILTER: memref<64x32x3x3xf32> = dense<2.0>
memref.global "private" @OUTPUT: memref<1x64x28x28xf32> = dense<0.0>

// ============================================================================
// 手工优化的 Conv2D 内核 (修复后)
// ============================================================================
func.func @kernel(%in: memref<1x32x30x30xf32>, %fil: memref<64x32x3x3xf32>, %out: memref<1x64x28x28xf32>) {
  %t_start = call @rtclock() : () -> f64

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c28 = arith.constant 28 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %f0 = arith.constant 0.0 : f32

  // --- 分块循环 ---
  affine.parallel (%oc0, %h0) = (0, 0) to (%c64, %c28) step (16, 7) {
    affine.for %w0 = 0 to 28 step 16 {
      affine.for %ic0 = 0 to 32 step 8 {
        affine.for %oc1 = 0 to 16 {
          %oc = arith.addi %oc0, %oc1 : index
          affine.for %h1 = 0 to 7 {
            %h = arith.addi %h0, %h1 : index

            // --- 向量化循环 (W 维度) ---
            affine.for %w1 = 0 to 16 step 8 {
              %w = arith.addi %w0, %w1 : index

              // --- 1. 计算掩码以处理向量化尾部 ---
              %w_limit = affine.min #map_w_bound(%w0)
              %remaining_w = arith.subi %w_limit, %w : index
              %vl = arith.minsi %remaining_w, %c8 : index
              %mask = vector.create_mask %vl : vector<8xi1>

              // --- 2. 准备累加器 (使用 transfer_read 和掩码) ---
              %acc_vec = vector.transfer_read %out[%c0, %oc, %h, %w], %f0, %mask
                {in_bounds = [false]}
                : memref<1x64x28x28xf32>, vector<8xf32>

              // --- 3. 归约循环 ---
              %ic0_sym = affine.apply affine_map<(d0) -> (d0)>(%ic0)
              %final_vec = affine.for %ic = #map_id(%ic0_sym) to #map_ic_bound(%ic0_sym) iter_args(%iter_acc1 = %acc_vec) -> vector<8xf32> {
                %res1 = affine.for %kh = 0 to 3 iter_args(%iter_acc2 = %iter_acc1) -> vector<8xf32> {
                  %res2 = affine.for %kw = 0 to 3 iter_args(%iter_acc3 = %iter_acc2) -> vector<8xf32> {

                    %ih = arith.addi %h, %kh : index
                    %iw = arith.addi %w, %kw : index

                    // --- 输入读取 (带掩码) ---
                    %in_vec = vector.transfer_read %in[%c0, %ic, %ih, %iw], %f0, %mask
                      {in_bounds = [false]}
                      : memref<1x32x30x30xf32>, vector<8xf32>

                    // --- 滤波器读取 (标量) ---
                    %fil_scalar = memref.load %fil[%oc, %ic, %kh, %kw] : memref<64x32x3x3xf32>
                    %fil_vec = vector.splat %fil_scalar : vector<8xf32>

                    // --- 累加器更新 ---
                    %next_acc = vector.fma %in_vec, %fil_vec, %iter_acc3 : vector<8xf32>
                    affine.yield %next_acc : vector<8xf32>
                  }
                  affine.yield %res2 : vector<8xf32>
                }
                affine.yield %res1 : vector<8xf32>
              }

              // --- 4. 存储结果 (使用 transfer_write 和掩码) ---
              vector.transfer_write %final_vec, %out[%c0, %oc, %h, %w], %mask
                {in_bounds = [false]}
                : vector<8xf32>, memref<1x64x28x28xf32>
            }
          }
        }
      }
    }
    affine.yield
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %out_cast = memref.cast %out : memref<1x64x28x28xf32> to memref<*xf32>
  call @printMemrefF32(%out_cast) : (memref<*xf32>) -> ()
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