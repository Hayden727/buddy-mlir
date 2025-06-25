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

# 基于 Tensor 的测试用例
test-simple-add:
	@echo "Testing simple element-wise addition..."
	@$(BUDDY_OPT) ./test-simple-add.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
	| $(BUDDY_OPT) \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts \
	| $(MLIR_CPU_RUNNER) -e main -entry-point-result=void \
		-shared-libs=$(MLIR_RUNNER_UTILS) \
		-shared-libs=$(MLIR_C_RUNNER_UTILS)

test-matrix-elementwise:
	@echo "Testing matrix element-wise operations..."
	@$(BUDDY_OPT) ./test-matrix-elementwise.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
	| $(BUDDY_OPT) \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts \
	| $(MLIR_CPU_RUNNER) -e main -entry-point-result=void \
		-shared-libs=$(MLIR_RUNNER_UTILS) \
		-shared-libs=$(MLIR_C_RUNNER_UTILS)

test-vector-dot-product:
	@echo "Testing vector dot product (reduction)..."
	@$(BUDDY_OPT) ./test-vector-dot-product.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
	| $(BUDDY_OPT) \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts \
	| $(MLIR_CPU_RUNNER) -e main -entry-point-result=void \
		-shared-libs=$(MLIR_RUNNER_UTILS) \
		-shared-libs=$(MLIR_C_RUNNER_UTILS)

test-matrix-row-sum:
	@echo "Testing matrix row sum (mixed iterators)..."
	@$(BUDDY_OPT) ./test-matrix-row-sum.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
	| $(BUDDY_OPT) \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts \
	| $(MLIR_CPU_RUNNER) -e main -entry-point-result=void \
		-shared-libs=$(MLIR_RUNNER_UTILS) \
		-shared-libs=$(MLIR_C_RUNNER_UTILS)

test-broadcast-add:
	@echo "Testing broadcast addition..."
	@$(BUDDY_OPT) ./test-broadcast-add.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
	| $(BUDDY_OPT) \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts \
	| $(MLIR_CPU_RUNNER) -e main -entry-point-result=void \
		-shared-libs=$(MLIR_RUNNER_UTILS) \
		-shared-libs=$(MLIR_C_RUNNER_UTILS)

test-tensor-3d-ops:
	@echo "Testing 3D tensor operations..."
	@$(BUDDY_OPT) ./test-tensor-3d-ops.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
	| $(BUDDY_OPT) \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts \
	| $(MLIR_CPU_RUNNER) -e main -entry-point-result=void \
		-shared-libs=$(MLIR_RUNNER_UTILS) \
		-shared-libs=$(MLIR_C_RUNNER_UTILS)

test-complex-computation:
	@echo "Testing complex computation..."
	@$(BUDDY_OPT) ./test-complex-computation.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
	| $(BUDDY_OPT) \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts \
	| $(MLIR_CPU_RUNNER) -e main -entry-point-result=void \
		-shared-libs=$(MLIR_RUNNER_UTILS) \
		-shared-libs=$(MLIR_C_RUNNER_UTILS)

# 基于 MemRef 的测试用例
test-memref-simple-add:
	@echo "Testing memref simple addition..."
	@$(BUDDY_OPT) ./test-memref-simple-add.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
	| $(BUDDY_OPT) \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-buffer-deallocation \
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
		-reconcile-unrealized-casts \
	| $(MLIR_CPU_RUNNER) -e main -entry-point-result=void \
		-shared-libs=$(MLIR_RUNNER_UTILS) \
		-shared-libs=$(MLIR_C_RUNNER_UTILS)

test-dynamic-dims:
	@echo "Testing dynamic dimensions..."
	@$(BUDDY_OPT) ./test-dynamic-dims.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
	| $(BUDDY_OPT) \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-buffer-deallocation \
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
		-reconcile-unrealized-casts \
	| $(MLIR_CPU_RUNNER) -e main -entry-point-result=void \
		-shared-libs=$(MLIR_RUNNER_UTILS) \
		-shared-libs=$(MLIR_C_RUNNER_UTILS)

# 分析模式测试（仅运行分析，不执行）
test-analysis-only:
	@echo "Running analysis-only tests..."
	@echo "=== Simple Add Analysis ==="
	@$(BUDDY_OPT) ./test-simple-add.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
		-mlir-print-ir-after-all
	@echo "=== Matrix Elementwise Analysis ==="
	@$(BUDDY_OPT) ./test-matrix-elementwise.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
		-mlir-print-ir-after-all
	@echo "=== Vector Dot Product Analysis ==="
	@$(BUDDY_OPT) ./test-vector-dot-product.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
		-mlir-print-ir-after-all
	@echo "=== Dynamic Dims Analysis ==="
	@$(BUDDY_OPT) ./test-dynamic-dims.mlir \
		-pass-pipeline "builtin.module(func.func(linalg-generic-optimization))" \
		-mlir-print-ir-after-all

# 运行所有基于Tensor的测试
test-all-tensor: test-simple-add test-matrix-elementwise test-vector-dot-product test-matrix-row-sum test-broadcast-add test-tensor-3d-ops test-complex-computation

# 运行所有基于MemRef的测试
test-all-memref: test-memref-simple-add test-dynamic-dims

# 运行所有测试
test-all: test-all-tensor test-all-memref

# 清理生成的文件
clean:
	@echo "Cleaning up..."
	@rm -f *.ll *.o *.so

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  test-simple-add          - Test simple element-wise addition"
	@echo "  test-matrix-elementwise  - Test matrix element-wise operations"
	@echo "  test-vector-dot-product  - Test vector dot product (reduction)"
	@echo "  test-matrix-row-sum      - Test matrix row sum (mixed iterators)"
	@echo "  test-broadcast-add       - Test broadcast addition"
	@echo "  test-tensor-3d-ops       - Test 3D tensor operations"
	@echo "  test-complex-computation - Test complex computation"
	@echo "  test-memref-simple-add   - Test memref simple addition"
	@echo "  test-dynamic-dims        - Test dynamic dimensions"
	@echo "  test-analysis-only       - Run analysis without execution"
	@echo "  test-all-tensor          - Run all tensor-based tests"
	@echo "  test-all-memref          - Run all memref-based tests"
	@echo "  test-all                 - Run all tests"
	@echo "  clean                    - Clean up generated files"
	@echo "  help                     - Show this help message"

.PHONY: test-simple-add test-matrix-elementwise test-vector-dot-product test-matrix-row-sum test-broadcast-add test-tensor-3d-ops test-complex-computation test-memref-simple-add test-dynamic-dims test-analysis-only test-all-tensor test-all-memref test-all clean help