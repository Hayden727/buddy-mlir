// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -tensor-bufferize \
// RUN:     -buffer-deallocation \
// RUN:     -finalizing-bufferize \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @rtclock() -> f64

  func.func @kernel_fpowi(%arg0: tensor<1x32x40x64xf32>) {
    %t_start = call @rtclock() : () -> f64

    // Power operation
    %c2_i32 = arith.constant 2 : i32
    %output_tensor = tensor.empty() : tensor<1x32x40x64xf32>
    %result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x32x40x64xf32>) outs(%output_tensor : tensor<1x32x40x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = math.fpowi %in, %c2_i32 : f32, i32
        linalg.yield %0 : f32
    } -> tensor<1x32x40x64xf32>

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64

    %tensor_unranked = tensor.cast %result : tensor<1x32x40x64xf32> to tensor<*xf32>

    // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 32, 40, 64] strides = [81920, 2560, 64, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SAME: [25{{(, 25)*}}],

    call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
    vector.print %time : f64

    return
  }

  func.func @main() {
    %input_tensor = arith.constant dense<5.0> : tensor<1x32x40x64xf32>

    call @kernel_fpowi(%input_tensor) : (tensor<1x32x40x64xf32>) -> ()

    return
  }

  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
}