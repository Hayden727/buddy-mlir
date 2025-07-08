// 外部函数声明
func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

memref.global "private" @A_large_opt: memref<256x1024xf32> = dense<2.0>
memref.global "private" @B_large_opt: memref<1024x512xf32> = dense<3.0>
memref.global "private" @C_large_opt: memref<256x512xf32> = dense<1.0>

func.func @kernel_large_opt(%a: memref<256x1024xf32>, %b: memref<1024x512xf32>, %c: memref<256x512xf32>) {
  %t_start = call @rtclock() : () -> f64
  affine.parallel (%m0, %n0) = (0, 0) to (256, 512) step (8, 32) {
    affine.for %m1 = 0 to 8 {
      %m = arith.addi %m0, %m1 : index
      affine.for %n1 = 0 to 32 step 8 {
        %n = arith.addi %n0, %n1 : index
        %c_vec = vector.load %c[%m, %n] : memref<256x512xf32>, vector<8xf32>
        %final_vec = affine.for %k = 0 to 1024 iter_args(%acc = %c_vec) -> vector<8xf32> {
          %a_s = memref.load %a[%m, %k] : memref<256x1024xf32>
          %a_v = vector.splat %a_s : vector<8xf32>
          %b_v = vector.load %b[%k, %n] : memref<1024x512xf32>, vector<8xf32>
          %next = vector.fma %a_v, %b_v, %acc : vector<8xf32>
          affine.yield %next : vector<8xf32>
        }
        vector.store %final_vec, %c[%m, %n] : memref<256x512xf32>, vector<8xf32>
      }
    }
  }
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  %c_cast = memref.cast %c : memref<256x512xf32> to memref<*xf32>
  call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()
  vector.print %time : f64
  return
}

func.func @main() { // 优化版本的入口
  %a = memref.get_global @A_large_opt : memref<256x1024xf32>
  %b = memref.get_global @B_large_opt : memref<1024x512xf32>
  %c = memref.get_global @C_large_opt : memref<256x512xf32>
  call @kernel_large_opt(%a, %b, %c) : (memref<256x1024xf32>, memref<1024x512xf32>, memref<256x512xf32>) -> ()
  return
}