#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
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

test-generic-gemm-run:
	@${MLIR_OPT} ./generic-gemm.mlir \
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
	@${MLIR_OPT} ./generic-gemm-manual.mlir \
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

test-generic-conv2d-run:
	@${MLIR_OPT} ./generic-conv2D.mlir \
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
	@${MLIR_OPT} ./generic-conv2D-manual.mlir \
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

test-generic-axpy-run:
	@${MLIR_OPT} ./generic-axpy.mlir \
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
	@${MLIR_OPT} ./generic-axpy-manual.mlir \
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

test-generic-dot-produce-run:
	@${MLIR_OPT} ./generic-dot-produce.mlir \
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

test-generic-dot-produce-manual-run:
	@${MLIR_OPT} ./generic-dot-produce-manual.mlir \
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

test-generic-batch-mat-vec-run:
	@${MLIR_OPT} ./generic-batch-mat-vec.mlir \
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
	@${MLIR_OPT} ./generic-batch-mat-vec-manual.mlir \
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

test-generic-outer-produce-run:
	@${MLIR_OPT} ./generic-outer-produce.mlir \
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

test-generic-outer-produce-manual-run:
	@${MLIR_OPT} ./generic-outer-produce-manual.mlir \
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

test-generic-softmax-run:
	@${MLIR_OPT} ./generic-softmax.mlir \
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
	@${MLIR_OPT} ./generic-softmax-manual.mlir \
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