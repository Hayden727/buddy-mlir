// ============================================================================
// GEMM - 极小规模 (Very Small) - 手工优化
// M=32, N=64, K=128
// ============================================================================

// 外部函数与全局内存声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

memref.global "private" @A_vsmall_opt: memref<32x128xf32> = dense<2.0>
memref.global "private" @B_vsmall_opt: memref<128x64xf32> = dense<3.0>
memref.global "private" @C_vsmall_opt: memref<32x64xf32> = dense<1.0>

// 手工优化的 GEMM 内核
func.func @kernel_vsmall_opt(%a: memref<32x128xf32>, %b: memref<128x64xf32>, %c: memref<32x64xf32>) {
  %t_start = call @rtclock() : () -> f64

  // 并行分块循环，范围改变，步长(分块大小)不变
  affine.parallel (%m0, %n0) = (0, 0) to (32, 64) step (8, 32) {
    // 块内循环范围不变
    affine.for %m1 = 0 to 8 {
      %m = arith.addi %m0, %m1 : index

      // 向量化计算循环，块内范围不变
      affine.for %n1 = 0 to 32 step 8 {
        %n = arith.addi %n0, %n1 : index

        %c_vec = vector.load %c[%m, %n] : memref<32x64xf32>, vector<8xf32>

        // k 维度归约循环，范围改变
        %final_vec = affine.for %k = 0 to 128 iter_args(%acc_vec = %c_vec) -> vector<8xf32> {
          %a_scalar = memref.load %a[%m, %k] : memref<32x128xf32>
          %a_vec = vector.splat %a_scalar : vector<8xf32>
          %b_vec = vector.load %b[%k, %n] : memref<128x64xf32>, vector<8xf32>
          %next_acc = vector.fma %a_vec, %b_vec, %acc_vec : vector<8xf32>
          affine.yield %next_acc : vector<8xf32>
        }
        vector.store %final_vec, %c[%m, %n] : memref<32x64xf32>, vector<8xf32>
      }
    }
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  %c_cast = memref.cast %c : memref<32x64xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()
  
  vector.print %time : f64
  
  return
}

// 主入口函数
func.func @main() {
  %A_mem = memref.get_global @A_vsmall_opt : memref<32x128xf32>
  %B_mem = memref.get_global @B_vsmall_opt : memref<128x64xf32>
  %C_mem = memref.get_global @C_vsmall_opt : memref<32x64xf32>

  call @kernel_vsmall_opt(%A_mem, %B_mem, %C_mem) : (memref<32x128xf32>, memref<128x64xf32>, memref<32x64xf32>) -> ()

  return
}