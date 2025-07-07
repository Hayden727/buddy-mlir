// 外部函数与全局内存声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

memref.global "private" @A: memref<128x512xf32> = dense<2.0>
memref.global "private" @B: memref<512x256xf32> = dense<3.0>
memref.global "private" @C: memref<128x256xf32> = dense<1.0>

// 手工优化的 GEMM 内核
func.func @kernel(%a: memref<128x512xf32>, %b: memref<512x256xf32>, %c: memref<128x256xf32>) {
  %t_start = call @rtclock() : () -> f64

  // 并行分块循环 (m, n 维度)
  affine.parallel (%m0, %n0) = (0, 0) to (128, 256) step (8, 32) {
    affine.for %m1 = 0 to 8 {
      %m = arith.addi %m0, %m1 : index

      // 向量化计算循环
      affine.for %n1 = 0 to 32 step 8 {
        %n = arith.addi %n0, %n1 : index

        %c_vec = vector.load %c[%m, %n] : memref<128x256xf32>, vector<8xf32>

        // k 维度归约循环
        // 注意：k的整个范围(0 to 512)被累加，没有分块
        %final_vec = affine.for %k = 0 to 512 iter_args(%acc_vec = %c_vec) -> vector<8xf32> {
          %a_scalar = memref.load %a[%m, %k] : memref<128x512xf32>
          %a_vec = vector.splat %a_scalar : vector<8xf32>

          %b_vec = vector.load %b[%k, %n] : memref<512x256xf32>, vector<8xf32>
          
          %next_acc = vector.fma %a_vec, %b_vec, %acc_vec : vector<8xf32>
          affine.yield %next_acc : vector<8xf32>
        }

        // FIX: vector.store 语法修正
        vector.store %final_vec, %c[%m, %n] : memref<128x256xf32> , vector<8xf32>
      }
    }
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 打印计算结果
  %c_cast = memref.cast %c : memref<128x256xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()
  
  // 打印时间
  vector.print %time : f64
  
  return
}

// 主入口函数
func.func @main() {
  %A_mem = memref.get_global @A : memref<128x512xf32>
  %B_mem = memref.get_global @B : memref<512x256xf32>
  %C_mem = memref.get_global @C : memref<128x256xf32>

  call @kernel(%A_mem, %B_mem, %C_mem) : (memref<128x512xf32>, memref<512x256xf32>, memref<128x256xf32>) -> ()

  return
}