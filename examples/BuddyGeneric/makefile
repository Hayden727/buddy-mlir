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
# Common Pass Pipeline Definition
# ===----------------------------------------------------------------------===//

# Define the common MLIR optimization pass pipeline
MLIR_PASS_PIPELINE := -arith-expand \
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
	-reconcile-unrealized-casts

# Define common MLIR CPU runner options
MLIR_CPU_RUNNER_OPTS := ${OPT_FLAG} -e main -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} \
	-shared-libs=${MLIR_C_RUNNER_UTILS}

# ===----------------------------------------------------------------------===//
# Common Test Functions
# ===----------------------------------------------------------------------===//

# Function to run C++ tests with g++
define run_cpp_g++_test
	@${G++_COMPILER} -O3 $(1) -o $(2)
	@$(2)
endef

# Function to run C++ tests with clang++
define run_cpp_clang++_test
	@${CLANG++_COMPILER} -O3 -march=native -ffast-math $(1) -o $(2)
	@$(2)
endef

# Function to run MLIR tests
define run_mlir_test
	@${MLIR_OPT} $(1) ${MLIR_PASS_PIPELINE} | \
	${MLIR_CPU_RUNNER} ${MLIR_CPU_RUNNER_OPTS}
endef

# ===----------------------------------------------------------------------===//
# Linalg Generic Optimization Tests
# ===----------------------------------------------------------------------===//

test-generic-gemm-g++-run:
	$(call run_cpp_g++_test,./GEMM/gemm.cpp,./GEMM/gemm_cpp_g++)

test-generic-gemm-clang++-run:
	$(call run_cpp_clang++_test,./GEMM/gemm.cpp,./GEMM/gemm_cpp_clang++)

test-generic-gemm-run:
	$(call run_mlir_test,./GEMM/generic-gemm.mlir)

test-generic-gemm-manual-run:
	$(call run_mlir_test,./GEMM/generic-gemm-manual.mlir)

test-generic-conv2d-g++-run:
	$(call run_cpp_g++_test,./Conv2D/conv2d.cpp,./Conv2D/conv2d_cpp_g++)

test-generic-conv2d-clang++-run:
	$(call run_cpp_clang++_test,./Conv2D/conv2d.cpp,./Conv2D/conv2d_cpp_clang++)

test-generic-conv2d-run:
	$(call run_mlir_test,./Conv2D/generic-conv2D.mlir)

test-generic-conv2d-manual-run:
	$(call run_mlir_test,./Conv2D/generic-conv2D-manual.mlir)

test-generic-axpy-g++-run:
	$(call run_cpp_g++_test,./AXPY/axpy.cpp,./AXPY/axpy_cpp_g++)

test-generic-axpy-clang++-run:
	$(call run_cpp_clang++_test,./AXPY/axpy.cpp,./AXPY/axpy_cpp_clang++)

test-generic-axpy-run:
	$(call run_mlir_test,./AXPY/generic-axpy.mlir)

test-generic-axpy-manual-run:
	$(call run_mlir_test,./AXPY/generic-axpy-manual.mlir)

test-generic-dot-product-g++-run:
	$(call run_cpp_g++_test,./Dot_Product/dot-product.cpp,./Dot_Product/dot-product_cpp_g++)

test-generic-dot-product-clang++-run:
	$(call run_cpp_clang++_test,./Dot_Product/dot-product.cpp,./Dot_Product/dot-product_cpp_clang++)

test-generic-dot-product-run:
	$(call run_mlir_test,./Dot_Product/generic-dot-product.mlir)

test-generic-dot-product-manual-run:
	$(call run_mlir_test,./Dot_Product/generic-dot-product-manual.mlir)

test-generic-batch-mat-vec-g++-run:
	$(call run_cpp_g++_test,./Batch_Matmul/batch-mat-vec.cpp,./Batch_Matmul/batch-mat-vec_cpp_g++)

test-generic-batch-mat-vec-clang++-run:
	$(call run_cpp_clang++_test,./Batch_Matmul/batch-mat-vec.cpp,./Batch_Matmul/batch-mat-vec_cpp_clang++)

test-generic-batch-mat-vec-run:
	$(call run_mlir_test,./Batch_Matmul/generic-batch-mat-vec.mlir)

test-generic-batch-mat-vec-manual-run:
	$(call run_mlir_test,./Batch_Matmul/generic-batch-mat-vec-manual.mlir)

test-generic-outer-product-g++-run:
	$(call run_cpp_g++_test,./Outer_Product/outer-product.cpp,./Outer_Product/outer-product_cpp_g++)

test-generic-outer-product-clang++-run:
	$(call run_cpp_clang++_test,./Outer_Product/outer-product.cpp,./Outer_Product/outer-product_cpp_clang++)

test-generic-outer-product-run:
	$(call run_mlir_test,./Outer_Product/generic-outer-product.mlir)

test-generic-outer-product-manual-run:
	$(call run_mlir_test,./Outer_Product/generic-outer-product-manual.mlir)

test-generic-softmax-g++-run:
	$(call run_cpp_g++_test,./Softmax/softmax.cpp,./Softmax/softmax_cpp_g++)

test-generic-softmax-clang++-run:
	$(call run_cpp_clang++_test,./Softmax/softmax.cpp,./Softmax/softmax_cpp_clang++)

test-generic-softmax-run:
	$(call run_mlir_test,./Softmax/generic-softmax.mlir)

test-generic-softmax-manual-run:
	$(call run_mlir_test,./Softmax/generic-softmax-manual.mlir)

test-generic-batch-gemm-g++-run:
	$(call run_cpp_g++_test,./Batch_GEMM/batch-gemm.cpp,./Batch_GEMM/batch-gemm_cpp_g++)

test-generic-batch-gemm-clang++-run:
	$(call run_cpp_clang++_test,./Batch_GEMM/batch-gemm.cpp,./Batch_GEMM/batch-gemm_cpp_clang++)

test-generic-batch-gemm-run:
	$(call run_mlir_test,./Batch_GEMM/generic-batch-gemm.mlir)

test-generic-batch-gemm-manual-run:
	$(call run_mlir_test,./Batch_GEMM/generic-batch-gemm-manual.mlir)

test-generic-gemm-s-g++-run:
	$(call run_cpp_g++_test,./GEMM/gemm-S.cpp,./GEMM/gemm-S_cpp_g++)

test-generic-gemm-s-clang++-run:
	$(call run_cpp_clang++_test,./GEMM/gemm-S.cpp,./GEMM/gemm-S_cpp_clang++)

test-generic-gemm-s-run:
	$(call run_mlir_test,./GEMM/generic-gemm-S.mlir)

test-generic-gemm-s-manual-run:
	$(call run_mlir_test,./GEMM/generic-gemm-S-manual.mlir)

test-generic-gemm-l-g++-run:
	$(call run_cpp_g++_test,./GEMM/gemm-L.cpp,./GEMM/gemm-L_cpp_g++)

test-generic-gemm-l-clang++-run:
	$(call run_cpp_clang++_test,./GEMM/gemm-L.cpp,./GEMM/gemm-L_cpp_clang++)

test-generic-gemm-l-run:
	$(call run_mlir_test,./GEMM/generic-gemm-L.mlir)

test-generic-gemm-l-manual-run:
	$(call run_mlir_test,./GEMM/generic-gemm-L-manual.mlir)

test-generic-gemm-tiny-g++-run:
	$(call run_cpp_g++_test,./GEMM/gemm-tiny.cpp,./GEMM/gemm-tiny_cpp_g++)

test-generic-gemm-tiny-clang++-run:
	$(call run_cpp_clang++_test,./GEMM/gemm-tiny.cpp,./GEMM/gemm-tiny_cpp_clang++)

test-generic-gemm-tiny-run:
	$(call run_mlir_test,./GEMM/generic-gemm-tiny.mlir)

test-generic-gemm-tiny-manual-run:
	$(call run_mlir_test,./GEMM/generic-gemm-tiny-manual.mlir)

test-generic-gemm-ml-g++-run:
	$(call run_cpp_g++_test,./GEMM/gemm-ML.cpp,./GEMM/gemm-ML_cpp_g++)

test-generic-gemm-ml-clang++-run:
	$(call run_cpp_clang++_test,./GEMM/gemm-ML.cpp,./GEMM/gemm-ML_cpp_clang++)

test-generic-gemm-ml-run:
	$(call run_mlir_test,./GEMM/generic-gemm-ML.mlir)

test-generic-gemm-ml-manual-run:
	$(call run_mlir_test,./GEMM/generic-gemm-ML-manual.mlir)
