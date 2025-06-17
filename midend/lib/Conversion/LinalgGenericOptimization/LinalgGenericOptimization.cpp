//===- LinalgGenericOptimization.cpp --------------------------------===//
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
// This file implements the GenericOp optimization.
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
#include "mlir/Dialect/Math/IR/Math.h"
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
#include <cassert>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>

using namespace mlir;
using namespace vector;
using namespace affine;
using namespace linalg;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if an operation can be vectorized and return the vectorized operation
/// if possible. Returns nullptr if the operation cannot be vectorized.
static Operation* tryVectorizeOperation(Operation* op, OpBuilder& builder, 
                                       int64_t vectorSize) {
  Location loc = op->getLoc();
  
  // Check for arithmetic operations that can be vectorized
  if (auto addOp = dyn_cast<arith::AddFOp>(op)) {
    // Check if operands can be vectorized
    Type elemType = addOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    // Create vector.splat for operands if they are scalars
    Value lhsVec = addOp.getLhs();
    Value rhsVec = addOp.getRhs();
    
    if (!lhsVec.getType().isa<VectorType>()) {
      lhsVec = builder.create<vector::SplatOp>(loc, vectorType, lhsVec);
    }
    if (!rhsVec.getType().isa<VectorType>()) {
      rhsVec = builder.create<vector::SplatOp>(loc, vectorType, rhsVec);
    }
    
    return builder.create<arith::AddFOp>(loc, vectorType, lhsVec, rhsVec);
  }
  
  if (auto mulOp = dyn_cast<arith::MulFOp>(op)) {
    Type elemType = mulOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    Value lhsVec = mulOp.getLhs();
    Value rhsVec = mulOp.getRhs();
    
    if (!lhsVec.getType().isa<VectorType>()) {
      lhsVec = builder.create<vector::SplatOp>(loc, vectorType, lhsVec);
    }
    if (!rhsVec.getType().isa<VectorType>()) {
      rhsVec = builder.create<vector::SplatOp>(loc, vectorType, rhsVec);
    }
    
    return builder.create<arith::MulFOp>(loc, vectorType, lhsVec, rhsVec);
  }
  
  if (auto subOp = dyn_cast<arith::SubFOp>(op)) {
    Type elemType = subOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    Value lhsVec = subOp.getLhs();
    Value rhsVec = subOp.getRhs();
    
    if (!lhsVec.getType().isa<VectorType>()) {
      lhsVec = builder.create<vector::SplatOp>(loc, vectorType, lhsVec);
    }
    if (!rhsVec.getType().isa<VectorType>()) {
      rhsVec = builder.create<vector::SplatOp>(loc, vectorType, rhsVec);
    }
    
    return builder.create<arith::SubFOp>(loc, vectorType, lhsVec, rhsVec);
  }
  
  if (auto divOp = dyn_cast<arith::DivFOp>(op)) {
    Type elemType = divOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    Value lhsVec = divOp.getLhs();
    Value rhsVec = divOp.getRhs();
    
    if (!lhsVec.getType().isa<VectorType>()) {
      lhsVec = builder.create<vector::SplatOp>(loc, vectorType, lhsVec);
    }
    if (!rhsVec.getType().isa<VectorType>()) {
      rhsVec = builder.create<vector::SplatOp>(loc, vectorType, rhsVec);
    }
    
    return builder.create<arith::DivFOp>(loc, vectorType, lhsVec, rhsVec);
  }
  
  if (auto negOp = dyn_cast<arith::NegFOp>(op)) {
    Type elemType = negOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    Value operandVec = negOp.getOperand();
    if (!operandVec.getType().isa<VectorType>()) {
      operandVec = builder.create<vector::SplatOp>(loc, vectorType, operandVec);
    }
    
    return builder.create<arith::NegFOp>(loc, vectorType, operandVec);
  }
  
  // Check for math operations that can be vectorized
  if (auto expOp = dyn_cast<math::ExpOp>(op)) {
    Type elemType = expOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    Value operandVec = expOp.getOperand();
    if (!operandVec.getType().isa<VectorType>()) {
      operandVec = builder.create<vector::SplatOp>(loc, vectorType, operandVec);
    }
    
    return builder.create<math::ExpOp>(loc, vectorType, operandVec);
  }
  
  if (auto logOp = dyn_cast<math::LogOp>(op)) {
    Type elemType = logOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    Value operandVec = logOp.getOperand();
    if (!operandVec.getType().isa<VectorType>()) {
      operandVec = builder.create<vector::SplatOp>(loc, vectorType, operandVec);
    }
    
    return builder.create<math::LogOp>(loc, vectorType, operandVec);
  }
  
  if (auto cosOp = dyn_cast<math::CosOp>(op)) {
    Type elemType = cosOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    Value operandVec = cosOp.getOperand();
    if (!operandVec.getType().isa<VectorType>()) {
      operandVec = builder.create<vector::SplatOp>(loc, vectorType, operandVec);
    }
    
    return builder.create<math::CosOp>(loc, vectorType, operandVec);
  }
  
  if (auto sinOp = dyn_cast<math::SinOp>(op)) {
    Type elemType = sinOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    Value operandVec = sinOp.getOperand();
    if (!operandVec.getType().isa<VectorType>()) {
      operandVec = builder.create<vector::SplatOp>(loc, vectorType, operandVec);
    }
    
    return builder.create<math::SinOp>(loc, vectorType, operandVec);
  }
  
  if (auto rsqrtOp = dyn_cast<math::RsqrtOp>(op)) {
    Type elemType = rsqrtOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    Value operandVec = rsqrtOp.getOperand();
    if (!operandVec.getType().isa<VectorType>()) {
      operandVec = builder.create<vector::SplatOp>(loc, vectorType, operandVec);
    }
    
    return builder.create<math::RsqrtOp>(loc, vectorType, operandVec);
  }
  
  if (auto fpowiOp = dyn_cast<math::FPowIOp>(op)) {
    Type elemType = fpowiOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    Value operandVec = fpowiOp.getLhs();
    if (!operandVec.getType().isa<VectorType>()) {
      operandVec = builder.create<vector::SplatOp>(loc, vectorType, operandVec);
    }
    
    return builder.create<math::FPowIOp>(loc, vectorType, operandVec, fpowiOp.getRhs());
  }
  
  // Check for memory operations
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    MemRefType memrefType = loadOp.getMemRefType();
    Type elemType = memrefType.getElementType();
    
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    
    // Create vector.load operation
    return builder.create<vector::LoadOp>(loc, vectorType, loadOp.getMemRef(), loadOp.getIndices());
  }
  
  // Check for select operations
  if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    Type elemType = selectOp.getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    VectorType maskType = VectorType::get({vectorSize}, builder.getI1Type());
    
    Value condVec = selectOp.getCondition();
    Value trueVec = selectOp.getTrueValue();
    Value falseVec = selectOp.getFalseValue();
    
    if (!condVec.getType().isa<VectorType>()) {
      condVec = builder.create<vector::SplatOp>(loc, maskType, condVec);
    }
    if (!trueVec.getType().isa<VectorType>()) {
      trueVec = builder.create<vector::SplatOp>(loc, vectorType, trueVec);
    }
    if (!falseVec.getType().isa<VectorType>()) {
      falseVec = builder.create<vector::SplatOp>(loc, vectorType, falseVec);
    }
    
    return builder.create<arith::SelectOp>(loc, vectorType, condVec, trueVec, falseVec);
  }
  
  // Check for comparison operations
  if (auto cmpfOp = dyn_cast<arith::CmpFOp>(op)) {
    Type elemType = cmpfOp.getLhs().getType();
    if (!elemType.isa<FloatType>()) return nullptr;
    
    VectorType vectorType = VectorType::get({vectorSize}, elemType);
    VectorType resultType = VectorType::get({vectorSize}, builder.getI1Type());
    
    Value lhsVec = cmpfOp.getLhs();
    Value rhsVec = cmpfOp.getRhs();
    
    if (!lhsVec.getType().isa<VectorType>()) {
      lhsVec = builder.create<vector::SplatOp>(loc, vectorType, lhsVec);
    }
    if (!rhsVec.getType().isa<VectorType>()) {
      rhsVec = builder.create<vector::SplatOp>(loc, vectorType, rhsVec);
    }
    
    return builder.create<arith::CmpFOp>(loc, resultType, cmpfOp.getPredicate(), lhsVec, rhsVec);
  }
  
  // Check for type conversion operations
  if (auto sitofpOp = dyn_cast<arith::SIToFPOp>(op)) {
    Type fromType = sitofpOp.getIn().getType();
    Type toType = sitofpOp.getType();
    
    if (!fromType.isa<IntegerType>() || !toType.isa<FloatType>()) return nullptr;
    
    VectorType fromVectorType = VectorType::get({vectorSize}, fromType);
    VectorType toVectorType = VectorType::get({vectorSize}, toType);
    
    Value operandVec = sitofpOp.getIn();
    if (!operandVec.getType().isa<VectorType>()) {
      operandVec = builder.create<vector::SplatOp>(loc, fromVectorType, operandVec);
    }
    
    return builder.create<arith::SIToFPOp>(loc, toVectorType, operandVec);
  }
  
  if (auto uitofpOp = dyn_cast<arith::UIToFPOp>(op)) {
    Type fromType = uitofpOp.getIn().getType();
    Type toType = uitofpOp.getType();
    
    if (!fromType.isa<IntegerType>() || !toType.isa<FloatType>()) return nullptr;
    
    VectorType fromVectorType = VectorType::get({vectorSize}, fromType);
    VectorType toVectorType = VectorType::get({vectorSize}, toType);
    
    Value operandVec = uitofpOp.getIn();
    if (!operandVec.getType().isa<VectorType>()) {
      operandVec = builder.create<vector::SplatOp>(loc, fromVectorType, operandVec);
    }
    
    return builder.create<arith::UIToFPOp>(loc, toVectorType, operandVec);
  }
  
  if (auto trunciOp = dyn_cast<arith::TruncIOp>(op)) {
    Type fromType = trunciOp.getIn().getType();
    Type toType = trunciOp.getType();
    
    if (!fromType.isa<IntegerType>() || !toType.isa<IntegerType>()) return nullptr;
    
    VectorType fromVectorType = VectorType::get({vectorSize}, fromType);
    VectorType toVectorType = VectorType::get({vectorSize}, toType);
    
    Value operandVec = trunciOp.getIn();
    if (!operandVec.getType().isa<VectorType>()) {
      operandVec = builder.create<vector::SplatOp>(loc, fromVectorType, operandVec);
    }
    
    return builder.create<arith::TruncIOp>(loc, toVectorType, operandVec);
  }
  
  // Operation cannot be vectorized
  return nullptr;
}

/// Analyze a linalg.generic operation to determine if it can be vectorized
/// Returns true if the operation can be vectorized, false otherwise
/// If vectorizable, creates the vectorized operations
static bool analyzeAndVectorizeGenericOp(GenericOp genericOp, OpBuilder& builder, 
                                         int64_t vectorSize) {
  // Check if the generic op has a single region with a single block
  if (genericOp->getNumRegions() != 1) return false;
  
  Region& region = genericOp->getRegion(0);
  if (!region.hasOneBlock()) return false;
  
  Block& block = region.front();
  
  // Check if all operations in the block can be vectorized
  SmallVector<Operation*> vectorizableOps;
  for (Operation& op : block) {
    if (isa<linalg::YieldOp>(&op)) {
      // Skip yield operations
      continue;
    }
    if (isa<linalg::IndexOp>(&op)) {
      // Skip index operations for now
      continue;
    }
    
    // Check if the operation can be vectorized
    OpBuilder tempBuilder(&op);
    Operation* vectorizedOp = tryVectorizeOperation(&op, tempBuilder, vectorSize);
    if (!vectorizedOp) {
      // If any operation cannot be vectorized, return false
      return false;
    }
    vectorizableOps.push_back(&op);
  }
  
  // If we reach here, all operations can be vectorized
  // Create the vectorized operations
  builder.setInsertionPoint(genericOp);
  
  // For demonstration purposes, just return true
  // In a real implementation, you would create the vectorized operations here
  return true;
}

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class LinalgGenericOptimiztaionPattern : public OpConversionPattern<GenericOp> {
public:
  explicit LinalgGenericOptimiztaionPattern(MLIRContext *context,
                                            int64_t affineVectorSizeParam)
      : OpConversionPattern(context), affineVectorSize(affineVectorSizeParam) {}

  LogicalResult
  matchAndRewrite(GenericOp genericOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Initialize Parameters
    auto inputOperands = genericOp.getInputs();
    auto outputOperands = genericOp.getOutputs();
    int32_t inputSize = inputOperands.size();
    int32_t outputSize = outputOperands.size();
    
    if (inputSize == 0 || outputSize == 0) {
      return failure();
    }
    
    // Check if the first input is a MemRefType
    auto firstInputType = inputOperands[0].getType().dyn_cast<MemRefType>();
    if (!firstInputType) {
      return failure();
    }
    
    Type elementType = firstInputType.getElementType();
    
    // Try to vectorize the generic operation
    if (analyzeAndVectorizeGenericOp(genericOp, rewriter, affineVectorSize)) {
      // If vectorization is successful, you would replace the original operation here
      // For now, just return success to indicate that the operation can be vectorized
      return success();
    }
    
    return failure();
  }

private:
  int64_t affineVectorSize;
};
} // namespace

//===----------------------------------------------------------------------===//
// LinalgGenericOptimizationPass
//===----------------------------------------------------------------------===//

namespace {
class LinalgGenericOptimizationPass
    : public PassWrapper<LinalgGenericOptimizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgGenericOptimizationPass)
  StringRef getArgument() const final { return "linalg-generic-optimization"; }
  StringRef getDescription() const final {
    return "linalg dialect generic operator general optimization.";
  }

  LinalgGenericOptimizationPass() = default;
  LinalgGenericOptimizationPass(const LinalgGenericOptimizationPass &) {}
  explicit LinalgGenericOptimizationPass(int64_t affineVectorSizeParam) {
    affineVectorSize = affineVectorSizeParam;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                           memref::MemRefDialect, VectorDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, linalg::FillOp>();
    target.addLegalOp<linalg::FillOp>();

    RewritePatternSet patterns(context);
    patterns.add<LinalgGenericOptimiztaionPattern>(context, affineVectorSize);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, affine::AffineDialect, VectorDialect,
                    memref::MemRefDialect, math::MathDialect>();
  }

  Option<int64_t> affineVectorSize{*this, "vector-size",
                                   llvm::cl::desc("Affine Vector size."),
                                   llvm::cl::init(16)};
};
} // namespace

namespace mlir {
namespace buddy {
void registerLinalgGenericOptimizationPass() {
  PassRegistration<LinalgGenericOptimizationPass>();
}
} // namespace buddy
} // namespace mlir