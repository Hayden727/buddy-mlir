//===- GenericOpTransposeVectorization.cpp ----------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the transpose optimization.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

using namespace mlir;
using namespace vector;
using namespace affine;

namespace {
class GenericOpTransposeVectorizationPattern
    : public OpConversionPattern<linalg::GenericOp> {
public:
  explicit GenericOpTransposeVectorizationPattern(MLIRContext *context,
                                                  int64_t affineVectorSizeParam)
      : OpConversionPattern(context), affineVectorSize(affineVectorSizeParam) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp genericOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 检查是否为转置操作
    SmallVector<int64_t> permutation;
    if (!isTransposeOp(genericOp, rewriter, permutation)) {
      return failure();
    }

    Value input = genericOp.getInputs()[0];
    Value output = genericOp.getOutputs()[0];
    auto loc = genericOp.getLoc();
    Type elementType = input.getType().cast<MemRefType>().getElementType();
    int64_t rank = input.getType().cast<MemRefType>().getRank();

    // 定义常量
    const Value index0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value indexVecSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(affineVectorSize));
    std::vector<AffineExpr> ds(rank);
    for (int i = 0; i < rank; i++) {
      ds[i] = rewriter.getAffineDimExpr(i);
    }
    const AffineExpr zeroAffine = rewriter.getAffineConstantExpr(0);
    const Value zeroElementType = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // 获取张量维度
    std::vector<Value> dims(rank);
    for (int i = 0; i < rank; i++) {
      dims[i] = rewriter.create<memref::DimOp>(loc, input, i);
    }

    // 处理向量尾部
    std::vector<Value> unalignedLengths(rank);
    std::vector<Value> upperBounds(rank);
    for (int i = 0; i < rank; i++) {
      unalignedLengths[i] = rewriter.create<affine::AffineApplyOp>(
          loc, AffineMap::get(1, 0, ds[0] % affineVectorSize), ValueRange{dims[i]});
      upperBounds[i] = rewriter.create<affine::AffineApplyOp>(
          loc, AffineMap::get(1, 0, ds[0].floorDiv(affineVectorSize) * affineVectorSize),
          ValueRange{dims[i]});
    }

    
    
    return success();
  }

private:
  int64_t affineVectorSize;

  // 检查是否为转置操作，并提取排列
  bool isTransposeOp(linalg::GenericOp genericOp, ConversionPatternRewriter &rewriter,
                     SmallVector<int64_t> &permutation) const {
    if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
      return false;
    }

    SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
    if (indexingMaps.size() != 2) {
      return false;
    }

    AffineMap inputMap = indexingMaps[0];
    AffineMap outputMap = indexingMaps[1];
    if (inputMap.getNumDims() != outputMap.getNumDims()) {
      return false;
    }

    int rank = inputMap.getNumDims();
    permutation.resize(rank);
    for (int i = 0; i < rank; i++) {
      AffineExpr outputExpr = outputMap.getResult(i);
      bool found = false;
      for (int j = 0; j < rank; j++) {
        if (inputMap.getResult(j) == outputExpr) {
          permutation[i] = j;
          found = true;
          break;
        }
      }
      if (!found) return false;
    }

    SmallVector<utils::IteratorType> iteratorTypes = genericOp.getIteratorTypesArray();
    for (auto type : iteratorTypes) {
      if (type != utils::IteratorType::parallel) {
        return false;
      }
    }

    Region &body = genericOp.getRegion();
    if (!body.hasOneBlock()) return false;
    Block &block = body.front();
    if (!llvm::hasSingleElement(block.getOperations())) return false;
    Operation &onlyOp = block.front();
    if (!isa<linalg::YieldOp>(onlyOp)) return false;

    return true;
  }
};
} // namespace

namespace {
class GenericOpTransposeVectorizationPass
    : public PassWrapper<GenericOpTransposeVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenericOpTransposeVectorizationPass)
  StringRef getArgument() const final { return "genericOp-transpose-vectorization"; }
  StringRef getDescription() const final { return "Transpose Optimization for any rank tensor."; }
  GenericOpTransposeVectorizationPass() = default;
  GenericOpTransposeVectorizationPass(const GenericOpTransposeVectorizationPass &) {}
  explicit GenericOpTransposeVectorizationPass(int64_t affineVectorSizeParam) {
    affineVectorSize = affineVectorSizeParam;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                           memref::MemRefDialect, VectorDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, linalg::FillOp>();
    RewritePatternSet patterns(context);
    patterns.add<GenericOpTransposeVectorizationPattern>(context, affineVectorSize);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> affineVectorSize{*this, "vector-size",
                                   llvm::cl::desc("Affine Vector size."),
                                   llvm::cl::init(16)};
};
} // namespace

namespace mlir {
namespace buddy {
void registerGenericOpTransposeVectorizationPass() {
  PassRegistration<GenericOpTransposeVectorizationPass>();
}
} // namespace buddy
} // namespace mlir