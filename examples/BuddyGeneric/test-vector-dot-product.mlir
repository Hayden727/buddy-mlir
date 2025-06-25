// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
// RUN: | buddy-opt \
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

func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

// ===----------------------------------------------------------------------===//
// 测试用例: 向量内积 - 测试reduction操作
// 
// 这个测试用例验证：
// 1. Reduction迭代器类型的正确识别
// 2. 标量输出的处理
// 3. 累积操作的分析
// 4. 向量化不适用场景的判断（reduction通常不适合向量化）
// ===----------------------------------------------------------------------===//

func.func @test_vector_dot_product() -> tensor<f32> {
  %t_start = call @rtclock() : () -> f64
  
  // 创建输入向量
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %input1 = tensor.splat %c2 : tensor<1024xf32>
  %input2 = tensor.splat %c3 : tensor<1024xf32>
  
  // 初始化累积器为0
  %c0 = arith.constant 0.0 : f32
  %init = tensor.from_elements %c0 : tensor<f32>
  
  // 执行向量内积：sum(a[i] * b[i])
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]
  } ins(%input1, %input2 : tensor<1024xf32>, tensor<1024xf32>) 
    outs(%init : tensor<f32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %mul = arith.mulf %in0, %in1 : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<f32>
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // 验证结果 (2.0 * 3.0 * 1024 = 6144.0)
  %tensor_unranked = tensor.cast %result : tensor<f32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64
  
  return %result : tensor<f32>
}

func.func @main() {
  %result = call @test_vector_dot_product() : () -> tensor<f32>
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
// CHECK-NEXT: [6144] 