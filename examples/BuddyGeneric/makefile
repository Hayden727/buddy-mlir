#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
G++_COMPILER := g++
CLANG++_COMPILER := ../../llvm/build/bin/clang++
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
LIB_OMP := ../../llvm/build/lib/libomp.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

# ===----------------------------------------------------------------------===//
# Linalg Generic Optimization Tests
# ===----------------------------------------------------------------------===//

test-generic-gemm-g++-run:
	@${G++_COMPILER} -O3 ./GEMM/gemm.cpp -o ./GEMM/gemm_cpp_g++
	@./GEMM/gemm_cpp_g++

test-generic-gemm-clang++-run:
	@${CLANG++_COMPILER} -O3 -march=native -ffast-math ./GEMM/gemm.cpp -o ./GEMM/gemm_cpp_clang++
	@./GEMM/gemm_cpp_clang++

test-generic-gemm-run:
	@${MLIR_OPT} ./GEMM/generic-gemm.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS}

test-generic-gemm-manual-run:
	@${MLIR_OPT} ./GEMM/generic-gemm-manual.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-conv2d-g++-run:
	@${G++_COMPILER} -O3 ./Conv2D/conv2d.cpp -o ./Conv2D/conv2d_cpp_g++
	@./Conv2D/conv2d_cpp_g++

test-generic-conv2d-clang++-run:
	@${CLANG++_COMPILER} -O3 -march=native -ffast-math ./Conv2D/conv2d.cpp -o ./Conv2D/conv2d_cpp_clang++
	@./Conv2D/conv2d_cpp_clang++

test-generic-conv2d-run:
	@${MLIR_OPT} ./Conv2D/generic-conv2D.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS}

test-generic-conv2d-manual-run:
	@${MLIR_OPT} ./Conv2D/generic-conv2D-manual.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS}

test-generic-axpy-g++-run:
	@${G++_COMPILER} -O3 ./AXPY/axpy.cpp -o ./AXPY/axpy_cpp_g++
	@./AXPY/axpy_cpp_g++

test-generic-axpy-clang++-run:
	@${CLANG++_COMPILER} -O3 -march=native -ffast-math ./AXPY/axpy.cpp -o ./AXPY/axpy_cpp_clang++
	@./AXPY/axpy_cpp_clang++

test-generic-axpy-run:
	@${MLIR_OPT} ./AXPY/generic-axpy.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-axpy-manual-run:
	@${MLIR_OPT} ./AXPY/generic-axpy-manual.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-dot-product-g++-run:
	@${G++_COMPILER} -O3 ./Dot_Product/dot-product.cpp -o ./Dot_Product/dot-product_cpp_g++
	@./Dot_Product/dot-product_cpp_g++

test-generic-dot-product-clang++-run:
	@${CLANG++_COMPILER} -O3 -march=native -ffast-math ./Dot_Product/dot-product.cpp -o ./Dot_Product/dot-product_cpp_clang++
	@./Dot_Product/dot-product_cpp_clang++

test-generic-dot-product-run:
	@${MLIR_OPT} ./Dot_Product/generic-dot-product.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-dot-product-manual-run:
	@${MLIR_OPT} ./Dot_Product/generic-dot-product-manual.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-batch-mat-vec-g++-run:
	@${G++_COMPILER} -O3 ./Batch_Matmul/batch-mat-vec.cpp -o ./Batch_Matmul/batch-mat-vec_cpp_g++
	@./Batch_Matmul/batch-mat-vec_cpp_g++

test-generic-batch-mat-vec-clang++-run:
	@${CLANG++_COMPILER} -O3 -march=native -ffast-math ./Batch_Matmul/batch-mat-vec.cpp -o ./Batch_Matmul/batch-mat-vec_cpp_clang++
	@./Batch_Matmul/batch-mat-vec_cpp_clang++

test-generic-batch-mat-vec-run:
	@${MLIR_OPT} ./Batch_Matmul/generic-batch-mat-vec.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-batch-mat-vec-manual-run:
	@${MLIR_OPT} ./Batch_Matmul/generic-batch-mat-vec-manual.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-outer-product-g++-run:
	@${G++_COMPILER} -O3 ./Outer_Product/outer-product.cpp -o ./Outer_Product/outer-product_cpp_g++
	@./Outer_Product/outer-product_cpp_g++

test-generic-outer-product-clang++-run:
	@${CLANG++_COMPILER} -O3 -march=native -ffast-math ./Outer_Product/outer-product.cpp -o ./Outer_Product/outer-product_cpp_clang++
	@./Outer_Product/outer-product_cpp_clang++

test-generic-outer-product-run:
	@${MLIR_OPT} ./Outer_Product/generic-outer-product.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-outer-product-manual-run:
	@${MLIR_OPT} ./Outer_Product/generic-outer-product-manual.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-softmax-g++-run:
	@${G++_COMPILER} -O3 ./Softmax/softmax.cpp -o ./Softmax/softmax_cpp_g++
	@./Softmax/softmax_cpp_g++

test-generic-softmax-clang++-run:
	@${CLANG++_COMPILER} -O3 -march=native -ffast-math ./Softmax/softmax.cpp -o ./Softmax/softmax_cpp_clang++
	@./Softmax/softmax_cpp_clang++

test-generic-softmax-run:
	@${MLIR_OPT} ./Softmax/generic-softmax.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-softmax-manual-run:
	@${MLIR_OPT} ./Softmax/generic-softmax-manual.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-batch-gemm-g++-run:
	@${G++_COMPILER} -O3 ./Batch_GEMM/batch-gemm.cpp -o ./Batch_GEMM/batch-gemm_cpp_g++
	@./Batch_GEMM/batch-gemm_cpp_g++

test-generic-batch-gemm-clang++-run:
	@${CLANG++_COMPILER} -O3 -march=native -ffast-math ./Batch_GEMM/batch-gemm.cpp -o ./Batch_GEMM/batch-gemm_cpp_clang++
	@./Batch_GEMM/batch-gemm_cpp_clang++

test-generic-batch-gemm-run:
	@${MLIR_OPT} ./Batch_GEMM/generic-batch-gemm.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 

test-generic-batch-gemm-manual-run:
	@${MLIR_OPT} ./Batch_GEMM/generic-batch-gemm-manual.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} 
