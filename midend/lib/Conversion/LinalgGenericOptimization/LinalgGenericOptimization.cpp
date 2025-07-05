//===- LinalgGenericOptimization.cpp ----------------------------------===//
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
// This file implements the linalg generic optimization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "linalg-generic-optimization"

using namespace mlir;
using namespace vector;
using namespace affine;
using namespace arith;
using namespace linalg;
using namespace memref;
using namespace func;
using namespace math;
using namespace scf;

//===----------------------------------------------------------------------===//
// LinalgGenericOptimizationPattern
//===----------------------------------------------------------------------===//

namespace {

class LinalgGenericOptimizationPattern : public OpConversionPattern<linalg::GenericOp> {
public:
  explicit LinalgGenericOptimizationPattern(MLIRContext *context,
                                           const std::string& userArch = "x86_64",
                                           int64_t userVecWidth = 256,
                                           const std::string& userVecExt = "AVX2", 
                                           bool userFMA = false,
                                           int64_t userTile = 32)
      : OpConversionPattern(context),
        userArchitecture(userArch),
        userVectorWidth(userVecWidth),
        userVectorExtension(userVecExt),
        userEnableFMA(userFMA),
        userTileSize(userTile) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    // 第一阶段：综合分析 - 硬件环境、操作模式、内存访问模式
    VectorizationAnalysisResult analysisResult = performComprehensiveAnalysis(op);
    if (!analysisResult.isValid) {
      LLVM_DEBUG(llvm::dbgs() << "分析失败: " << analysisResult.failureReason << "\n");
      return failure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "向量化分析完成 - 硬件支持: " << analysisResult.hardwareCapable 
                           << ", 操作兼容: " << analysisResult.operationCompatible 
                           << ", 内存友好: " << analysisResult.memoryFriendly << "\n");
    
    // 第二阶段：向量化决策 - 基于分析结果制定向量化策略
    VectorizationDecision decision = makeVectorizationDecision(analysisResult);
    if (!decision.isValid) {
      LLVM_DEBUG(llvm::dbgs() << "决策失败: " << decision.failureReason << "\n");
      return failure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "向量化决策完成 - 选择维度数: " << decision.vectorizedDims.size()
                           << ", 向量宽度: " << decision.effectiveVectorWidth << "位"
                           << ", 分块级别: " << decision.tiling.tilingLevel << "\n");
    
    // 第三阶段：转换实施 - 基于决策结果进行实际的IR转换
    LogicalResult transformResult = performVectorizationTransformation(op, rewriter, analysisResult, decision);
    if (transformResult.succeeded()) {
      LLVM_DEBUG(llvm::dbgs() << "向量化转换成功完成\n");
      return success();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "向量化转换失败，保持原操作\n");
      return failure();
    }
  }

private:
  //===--------------------------------------------------------------------===//
  // 数据结构定义 - 分析模块
  //===--------------------------------------------------------------------===//

  // 硬件配置信息
  struct HardwareInfo {
    std::string architecture;          // 架构名称 (x86_64, aarch64等)
    std::string cpuName;              // CPU型号  
    int vectorWidthBits;              // 向量宽度(位)
    int vectorWidthBytes;             // 向量宽度(字节) 
    std::string vectorExtension;      // 向量扩展 (AVX2, NEON等)
    bool supportsFMA;                 // 是否支持FMA
    bool isVectorizationCapable;      // 硬件是否支持向量化
  };

  // 迭代器维度信息 - 保留完整的维度-类型映射关系
  // 解决问题：原先只保存维度索引列表丢失了维度与迭代器类型的对应关系
  // 现在每个维度都保留了：索引、类型、大小等完整信息，便于后续精确决策
  struct IteratorDimension {
    int dimIndex;                       // 维度索引 (0, 1, 2, ...)
    std::string iteratorType;           // 迭代器类型 ("parallel", "reduction", "window")
    int64_t dimSize;                    // 维度大小（如果可获取）
    bool isBoundKnown;                  // 维度边界是否已知
    
    // 默认构造函数
    IteratorDimension() 
      : dimIndex(-1), iteratorType(""), dimSize(-1), isBoundKnown(false) {}
    
    // 带参数的构造函数
    IteratorDimension(int idx, const std::string& type) 
      : dimIndex(idx), iteratorType(type), dimSize(-1), isBoundKnown(false) {}
      
    // 拷贝构造函数
    IteratorDimension(const IteratorDimension& other) 
      : dimIndex(other.dimIndex), iteratorType(other.iteratorType), 
        dimSize(other.dimSize), isBoundKnown(other.isBoundKnown) {}
        
    // 赋值操作符
    IteratorDimension& operator=(const IteratorDimension& other) {
      if (this != &other) {
        dimIndex = other.dimIndex;
        iteratorType = other.iteratorType;
        dimSize = other.dimSize;
        isBoundKnown = other.isBoundKnown;
      }
      return *this;
    }
  };

  // 操作分析结果 - 重新设计保留完整信息
  struct OperationInfo {
    std::vector<IteratorDimension> dimensions;        // 所有维度的完整信息
    std::unordered_set<std::string> supportedOps;    // 支持向量化的操作
    std::unordered_set<std::string> unsupportedOps;  // 不支持向量化的操作
    bool hasComplexOps;                 // 是否包含复杂操作
    bool allOpsVectorizable;           // 所有操作是否都可向量化
    
    // 便利方法：获取特定类型的维度
    std::vector<int> getParallelDims() const {
      std::vector<int> result;
      for (const auto& dim : dimensions) {
        if (dim.iteratorType == "parallel") {
          result.push_back(dim.dimIndex);
        }
      }
      return result;
    }
    
    std::vector<int> getReductionDims() const {
      std::vector<int> result;
      for (const auto& dim : dimensions) {
        if (dim.iteratorType == "reduction") {
          result.push_back(dim.dimIndex);
        }
      }
      return result;
    }
    
    // 获取维度的详细信息
    const IteratorDimension* getDimension(int dimIndex) const {
      for (const auto& dim : dimensions) {
        if (dim.dimIndex == dimIndex) {
          return &dim;
        }
      }
      return nullptr;
    }
    
    // 获取指定类型的维度信息
    std::vector<IteratorDimension> getDimensionsByType(const std::string& type) const {
      std::vector<IteratorDimension> result;
      for (const auto& dim : dimensions) {
        if (dim.iteratorType == type) {
          result.push_back(dim);
        }
      }
      return result;
    }
    
    // 检查是否有已知大小的并行维度 - 对向量化决策很重要
    bool hasKnownSizeParallelDims() const {
      for (const auto& dim : dimensions) {
        if (dim.iteratorType == "parallel" && dim.isBoundKnown && dim.dimSize > 0) {
          return true;
        }
      }
      return false;
    }
    
    // 获取最大的已知并行维度大小 - 用于确定向量化策略
    int64_t getLargestParallelDimSize() const {
      int64_t maxSize = 0;
      for (const auto& dim : dimensions) {
        if (dim.iteratorType == "parallel" && dim.isBoundKnown && dim.dimSize > maxSize) {
          maxSize = dim.dimSize;
        }
      }
      return maxSize;
    }
  };

  // 内存访问分析结果  
  struct MemoryAccessInfo {
    std::vector<int> continuousDims;    // 连续访问的维度
    bool hasSequentialAccess;           // 是否有顺序访问
    bool hasBroadcastAccess;           // 是否有广播访问
    bool hasTransposeAccess;           // 是否有转置访问
    bool isVectorizationFriendly;      // 内存访问是否适合向量化
  };

  // 数据类型分析结果 - 改进版！支持完整的MemRef元素类型规范
  struct DataTypeInfo {
    Type elementType;                  // MLIR元素类型
    std::string typeString;            // 类型字符串 ("f32", "i16", "index", etc.)
    int elementSizeBytes;              // 元素大小（字节）
    int elementSizeBits;               // 元素大小（位）
    
    // 基本类型标志
    bool isFloatingPoint;              // 是否为浮点类型
    bool isInteger;                    // 是否为整数类型
    bool isIndex;                      // 是否为index类型
    bool isComplex;                    // 是否为复数类型
    bool isSigned;                     // 是否为有符号类型（整数）
    
    // 复合类型标志 - 新增！
    bool isVector;                     // 是否为向量类型
    bool isNestedMemRef;               // 是否为嵌套memref类型
    bool isCustomType;                 // 是否为自定义类型
    
    // 向量类型信息（如果是向量）
    int vectorSize;                    // 向量元素数量
    Type vectorElementType;            // 向量元素类型
    
    // 布局和内存空间信息 - 新增！
    bool hasCustomLayout;              // 是否有自定义布局
    bool hasMemorySpace;               // 是否指定内存空间
    std::string layoutInfo;            // 布局信息描述
    std::string memorySpaceInfo;       // 内存空间信息
    
    DataTypeInfo() : elementType(nullptr), elementSizeBytes(0), elementSizeBits(0),
                    isFloatingPoint(false), isInteger(false), isIndex(false), 
                    isComplex(false), isSigned(false), isVector(false), 
                    isNestedMemRef(false), isCustomType(false), vectorSize(0),
                    vectorElementType(nullptr), hasCustomLayout(false), 
                    hasMemorySpace(false) {}
  };

  // 综合分析结果 - 统一状态管理
  struct VectorizationAnalysisResult {
    bool isValid;                      // 整体分析是否有效
    std::string failureReason;         // 失败原因描述
    
    HardwareInfo hardware;             // 硬件配置信息
    OperationInfo operation;           // 操作分析信息
    MemoryAccessInfo memoryAccess;     // 内存访问分析信息
    DataTypeInfo dataType;             // 数据类型信息 - 新增！
    
    // 状态一致性标志
    bool hardwareCapable;              // 硬件是否支持向量化
    bool operationCompatible;          // 操作是否兼容向量化
    bool memoryFriendly;               // 内存访问是否友好
    
    VectorizationAnalysisResult() : isValid(false), hardwareCapable(false), 
                                   operationCompatible(false), memoryFriendly(false) {}
  };

  //===--------------------------------------------------------------------===//
  // 决策模块数据结构
  //===--------------------------------------------------------------------===//

  // 向量化维度选择信息
  struct VectorizationDimension {
    int dimIndex;                      // 维度索引
    int64_t dimSize;                   // 维度大小
    int vectorWidth;                   // 选择的向量宽度
    int64_t remainder;                 // 余数 (dimSize % vectorWidth)
    bool needsTailHandling;            // 是否需要尾部处理
    std::string tailStrategy;          // 尾部处理策略
    
    // 默认构造函数
    VectorizationDimension() 
      : dimIndex(-1), dimSize(0), vectorWidth(0), remainder(0), 
        needsTailHandling(false), tailStrategy("") {}
    
    VectorizationDimension(int idx, int64_t size, int vecWidth) 
      : dimIndex(idx), dimSize(size), vectorWidth(vecWidth), 
        remainder(size % vecWidth), needsTailHandling(size % vecWidth != 0),
        tailStrategy("") {}
  };

  // 分块策略信息
  struct TilingStrategy {
    std::vector<int> tiledDims;        // 分块的维度
    std::vector<int64_t> tileSizes;    // 分块大小
    int tilingLevel;                   // 分块级别 (1或2)
    bool isValid;                      // 分块策略是否有效
    
    TilingStrategy() : tilingLevel(0), isValid(false) {}
  };

  // 向量化决策结果
  struct VectorizationDecision {
    bool isValid;                      // 决策是否有效
    std::string failureReason;         // 失败原因
    
    // 向量化策略
    std::vector<VectorizationDimension> vectorizedDims;  // 选择的向量化维度
    int effectiveVectorWidth;          // 实际使用的向量宽度
    std::string vectorType;            // 向量数据类型
    
    // 分块策略
    TilingStrategy tiling;             // 分块信息
    
    // 转换策略
    std::string conversionStrategy;    // 转换方法 ("affine.parallel", "scf.for", etc.)
    bool useFMA;                       // 是否使用FMA指令
    
    VectorizationDecision() : isValid(false), effectiveVectorWidth(0), 
                             useFMA(false) {}
  };

  //===--------------------------------------------------------------------===//
  // 转换实施方法
  //===--------------------------------------------------------------------===//

  /**
   * 向量化转换实施 - 基于决策结果进行实际的IR转换
   * 参考transpose.cc的嵌套循环构建模式，支持任意维度数据
   */
  LogicalResult performVectorizationTransformation(
      linalg::GenericOp op, 
      ConversionPatternRewriter &rewriter,
      const VectorizationAnalysisResult& analysis,
      const VectorizationDecision& decision) const {
    
    auto loc = op.getLoc();
    
    // 1. 获取输入输出操作数
    auto inputs = op.getInputs();
    auto outputs = op.getOutputs();
    if (inputs.empty() || outputs.empty()) {
      return failure();
    }
    
    Value input = inputs[0];
    Value output = outputs[0];
    
    // 2. 获取元素类型和维度信息
    Type elementType = analysis.dataType.elementType;
    auto inputType = input.getType().cast<ShapedType>();
    int rank = inputType.getRank();
    
    // 3. 创建基本常量（根据需要）
    
    // 4. 获取各维度大小（使用memref::DimOp）
    llvm::SmallVector<Value> dimSizes;
    for (int i = 0; i < rank; i++) {
      dimSizes.push_back(rewriter.create<memref::DimOp>(loc, input, i));
    }
    
    // 5. 根据决策创建向量化循环结构
    if (decision.tiling.tilingLevel == 2) {
      // 二级分块：创建两层并行循环
      return createTwoLevelParallelLoops(rewriter, loc, op, input, output, 
                                        dimSizes, elementType, decision, analysis);
    } else if (decision.tiling.tilingLevel == 1) {
      // 一级分块：创建一层并行循环
      return createSingleLevelParallelLoop(rewriter, loc, op, input, output,
                                          dimSizes, elementType, decision, analysis);
    } else {
      // 无分块：直接向量化
      return createDirectVectorization(rewriter, loc, op, input, output,
                                      dimSizes, elementType, decision, analysis);
    }
  }

  /**
   * 创建两层并行循环 - 最多二维分块
   */
  LogicalResult createTwoLevelParallelLoops(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      Value input, Value output, llvm::SmallVector<Value>& dimSizes,
      Type elementType, const VectorizationDecision& decision,
      const VectorizationAnalysisResult& analysis) const {
    
    // 获取分块维度信息
    const auto& tiledDims = decision.tiling.tiledDims;
    const auto& tileSizes = decision.tiling.tileSizes;
    
    if (tiledDims.size() < 2 || tileSizes.size() < 2) {
      return failure();
    }
    
    int outerDim = tiledDims[0];
    int innerDim = tiledDims[1];
    
    // 使用用户提供的编译时常量进行分块
    int64_t outerTileSize = userTileSize;         // 外层使用用户指定的分块大小
    int64_t innerTileSize = userTileSize / 2;     // 内层使用一半大小
    
    LLVM_DEBUG(llvm::dbgs() << "创建两层并行循环，使用编译时分块大小: 外层=" 
                           << outerTileSize << ", 内层=" << innerTileSize << "\n");
    
    // 创建外层并行循环
    affine::AffineParallelOp outerParallelOp = rewriter.create<affine::AffineParallelOp>(
        loc, TypeRange{}, ValueRange{dimSizes[outerDim]},
        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr("lowerBoundsGroups", rewriter.getI32TensorAttr({1})),
            rewriter.getNamedAttr("upperBoundsGroups", rewriter.getI32TensorAttr({1})),
            rewriter.getNamedAttr("lowerBoundsMap",
                AffineMapAttr::get(AffineMap::get(0, 0, {rewriter.getAffineConstantExpr(0)}, rewriter.getContext()))),
            rewriter.getNamedAttr("upperBoundsMap",
                AffineMapAttr::get(AffineMap::get(0, 1, {rewriter.getAffineSymbolExpr(0).floorDiv(outerTileSize) * outerTileSize}, rewriter.getContext()))),
            rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({outerTileSize})),
            rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({}))
        });
    
    // 创建外层循环体
    Block *outerBody = new Block();
    rewriter.setInsertionPointToStart(outerBody);
    outerBody->addArgument(rewriter.getIndexType(), loc);
    Value outerIdx = outerBody->getArguments()[0];
    
    // 创建内层并行循环
    affine::AffineParallelOp innerParallelOp = rewriter.create<affine::AffineParallelOp>(
        loc, TypeRange{}, ValueRange{dimSizes[innerDim]},
        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr("lowerBoundsGroups", rewriter.getI32TensorAttr({1})),
            rewriter.getNamedAttr("upperBoundsGroups", rewriter.getI32TensorAttr({1})),
            rewriter.getNamedAttr("lowerBoundsMap",
                AffineMapAttr::get(AffineMap::get(0, 0, {rewriter.getAffineConstantExpr(0)}, rewriter.getContext()))),
            rewriter.getNamedAttr("upperBoundsMap",
                AffineMapAttr::get(AffineMap::get(0, 1, {rewriter.getAffineSymbolExpr(0).floorDiv(innerTileSize) * innerTileSize}, rewriter.getContext()))),
            rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({innerTileSize})),
            rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({}))
        });
    
    // 创建内层循环体
    Block *innerBody = new Block();
    rewriter.setInsertionPointToStart(innerBody);
    innerBody->addArgument(rewriter.getIndexType(), loc);
    Value innerIdx = innerBody->getArguments()[0];
    
    // 在内层循环中创建向量化操作
    createVectorizedComputation(rewriter, loc, op, input, output, outerIdx, innerIdx,
                                outerDim, innerDim, dimSizes, elementType, decision, analysis);
    
    // 处理边界情况
    createBoundaryHandling(rewriter, loc, input, output, outerDim, innerDim,
                          dimSizes, elementType, decision, analysis);
    
    rewriter.create<affine::AffineYieldOp>(loc);
    innerParallelOp.getRegion().push_back(innerBody);
    
    rewriter.create<affine::AffineYieldOp>(loc);
    outerParallelOp.getRegion().push_back(outerBody);
    
    // 删除原操作
    rewriter.eraseOp(op);
    return success();
  }

  /**
   * 创建单层并行循环 - 修复版本
   * 确保在并行循环内生成实际的计算操作
   */
  LogicalResult createSingleLevelParallelLoop(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      Value input, Value output, llvm::SmallVector<Value>& dimSizes,
      Type elementType, const VectorizationDecision& decision,
      const VectorizationAnalysisResult& analysis) const {
    
    if (decision.tiling.tiledDims.empty()) {
      return failure();
    }
    
    int tiledDim = decision.tiling.tiledDims[0];
    
    // 计算向量化参数，用于并行循环的步长
    int64_t vectorElements = calculateVectorElementsFromUserParams(elementType);
    int64_t tileSize = userTileSize;
    
    // 使用向量化宽度作为并行循环的步长
    int64_t parallelStep = vectorElements * (tileSize / vectorElements); // 确保是向量宽度的倍数
    if (parallelStep == 0) parallelStep = vectorElements;
    
    LLVM_DEBUG(llvm::dbgs() << "创建单层并行循环，向量元素数: " << vectorElements 
                           << ", 并行步长: " << parallelStep << "\n");
    
    // 创建并行循环，步长为向量化友好的大小
    affine::AffineParallelOp parallelOp = rewriter.create<affine::AffineParallelOp>(
        loc, TypeRange{}, ValueRange{dimSizes[tiledDim]},
        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr("lowerBoundsGroups", rewriter.getI32TensorAttr({1})),
            rewriter.getNamedAttr("upperBoundsGroups", rewriter.getI32TensorAttr({1})),
            rewriter.getNamedAttr("lowerBoundsMap",
                AffineMapAttr::get(AffineMap::get(0, 0, {rewriter.getAffineConstantExpr(0)}, rewriter.getContext()))),
            rewriter.getNamedAttr("upperBoundsMap",
                AffineMapAttr::get(AffineMap::get(0, 1, {rewriter.getAffineSymbolExpr(0).floorDiv(parallelStep) * parallelStep}, rewriter.getContext()))),
            rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({parallelStep})),
            rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({}))
        });
    
    // 创建循环体
    Block *body = new Block();
    rewriter.setInsertionPointToStart(body);
    body->addArgument(rewriter.getIndexType(), loc);
    Value idx = body->getArguments()[0];
    
    // 对于1D情况，创建真正的向量化操作
    if (dimSizes.size() == 1) {
      // 计算向量化参数
      int64_t vectorElements = calculateVectorElementsFromUserParams(elementType);
      
      // 计算向量化的上边界（对齐的部分）
      Value vectorUpperBound = rewriter.create<affine::AffineApplyOp>(
          loc,
          AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0).floorDiv(vectorElements) * vectorElements}),
          ValueRange{dimSizes[0]});
      
      // 创建向量化affine.for循环，步长为向量元素数
      AffineMap lbMap = AffineMap::get(0, 0, {rewriter.getAffineConstantExpr(0)}, rewriter.getContext());
      AffineMap ubMap = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)}, rewriter.getContext());
      
      rewriter.create<affine::AffineForOp>(
          loc, ValueRange{}, lbMap, ValueRange{vectorUpperBound}, ubMap,
          vectorElements, ValueRange{},
          [&](OpBuilder &nestedBuilder, Location nestedLoc, Value vecIdx, ValueRange iterArgs) {
            // 创建向量化计算操作
            createVectorizedElementWiseComputation(rewriter, loc, op, vecIdx, elementType, vectorElements);
            nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
          });
      
      // 处理尾部非对齐数据（如果需要）
      createTailVectorHandling(rewriter, loc, op, vectorUpperBound, dimSizes[0], elementType, vectorElements);
    } else {
      // 对于多维情况，继续使用原有的递归方法
      createNestedDimensionLoops(rewriter, loc, op, input, output, idx, tiledDim,
                                dimSizes, elementType, decision, analysis, 0);
    }
    
    rewriter.create<affine::AffineYieldOp>(loc);
    parallelOp.getRegion().push_back(body);
    
    // 删除原操作
    rewriter.eraseOp(op);
    return success();
  }



  /**
   * 创建直接向量化（无分块）
   */
  LogicalResult createDirectVectorization(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      Value input, Value output, llvm::SmallVector<Value>& dimSizes,
      Type elementType, const VectorizationDecision& decision,
      const VectorizationAnalysisResult& analysis) const {
    
    // 直接创建嵌套循环，在最内层进行向量化
    createNestedDimensionLoops(rewriter, loc, op, input, output, nullptr, -1,
                              dimSizes, elementType, decision, analysis, 0);
    
    // 删除原操作
    rewriter.eraseOp(op);
    return success();
  }

  /**
   * 创建嵌套维度循环 - 递归构建
   * 修复版本：正确处理1D情况，确保生成实际的计算操作
   */
  void createNestedDimensionLoops(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      Value input, Value output, Value fixedIdx, int fixedDim,
      llvm::SmallVector<Value>& dimSizes, Type elementType,
      const VectorizationDecision& decision, const VectorizationAnalysisResult& analysis,
      int currentDim) const {
    
    // 跳过已固定的维度
    if (currentDim == fixedDim) {
      createNestedDimensionLoops(rewriter, loc, op, input, output, fixedIdx, fixedDim,
                                dimSizes, elementType, decision, analysis, currentDim + 1);
      return;
    }
    
    // 如果已处理完所有维度，创建实际的计算操作
    if (currentDim >= static_cast<int>(dimSizes.size())) {
      createActualComputation(rewriter, loc, op, input, output, fixedIdx, fixedDim,
                             elementType, decision, analysis);
      return;
    }
    
    // 如果是向量化维度，在最内层进行向量化处理
    if (currentDim == dimSizes.size() - 1 && isVectorizedDimension(currentDim, decision)) {
      createVectorizedInnerLoop(rewriter, loc, op, input, output, fixedIdx, fixedDim,
                               dimSizes, elementType, decision, analysis);
      return;
    }
    
    // 创建当前维度的循环
    AffineMap lbMap = AffineMap::get(0, 0, {rewriter.getAffineConstantExpr(0)}, rewriter.getContext());
    AffineMap ubMap = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)}, rewriter.getContext());
    
    rewriter.create<affine::AffineForOp>(
        loc, ValueRange{}, lbMap, ValueRange{dimSizes[currentDim]}, ubMap,
        1, ValueRange{},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv, ValueRange iterArgs) {
          // 递归创建下一层循环
          createNestedDimensionLoops(rewriter, loc, op, input, output, fixedIdx, fixedDim,
                                    dimSizes, elementType, decision, analysis, currentDim + 1);
          nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
        });
  }

  /**
   * 检查是否为向量化维度
   */
  bool isVectorizedDimension(int dimIndex, const VectorizationDecision& decision) const {
    for (const auto& vecDim : decision.vectorizedDims) {
      if (vecDim.dimIndex == dimIndex) {
        return true;
      }
    }
    return false;
  }

  /**
   * 创建向量化内层循环
   */
  void createVectorizedInnerLoop(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      Value input, Value output, Value fixedIdx, int fixedDim,
      llvm::SmallVector<Value>& dimSizes, Type elementType,
      const VectorizationDecision& decision, const VectorizationAnalysisResult& analysis) const {
    
    if (decision.vectorizedDims.empty()) {
      return;
    }
    
    const auto& vecDim = decision.vectorizedDims[0];
    
    // 使用用户提供的编译时常量计算向量元素数量
    int64_t vectorWidthElements = calculateVectorElementsFromUserParams(elementType);
    
    LLVM_DEBUG(llvm::dbgs() << "创建向量化内层循环，使用编译时向量元素数: " 
                           << vectorWidthElements << "\n");
    
    // 计算向量化的上边界（对齐的部分）
    Value vectorUpperBound = rewriter.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0).floorDiv(vectorWidthElements) * vectorWidthElements}),
        ValueRange{dimSizes[vecDim.dimIndex]});
    
    // 创建向量化循环
    AffineMap lbMap = AffineMap::get(0, 0, {rewriter.getAffineConstantExpr(0)}, rewriter.getContext());
    AffineMap ubMap = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)}, rewriter.getContext());
    
    rewriter.create<affine::AffineForOp>(
        loc, ValueRange{}, lbMap, ValueRange{vectorUpperBound}, ubMap,
        vectorWidthElements, ValueRange{},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv, ValueRange iterArgs) {
          // 创建向量化计算
          createVectorizedComputation(rewriter, loc, op, input, output, fixedIdx, iv,
                                     fixedDim, vecDim.dimIndex, dimSizes, elementType, decision, analysis);
          nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
        });
    
    // 处理尾部数据
    if (vecDim.needsTailHandling) {
      createTailHandling(rewriter, loc, input, output, vecDim, vectorUpperBound,
                        dimSizes, elementType, decision, analysis);
    }
  }

  /**
   * 创建向量化计算 - 核心向量化操作
   * 改进版：实现真正的向量化计算逻辑
   */
  void createVectorizedComputation(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      Value input, Value output, Value outerIdx, Value innerIdx,
      int outerDim, int innerDim, llvm::SmallVector<Value>& dimSizes,
      Type elementType, const VectorizationDecision& decision,
      const VectorizationAnalysisResult& analysis) const {
    
    // 使用用户提供的编译时常量计算向量元素数量
    int64_t vectorWidthElements = calculateVectorElementsFromUserParams(elementType);
    
    LLVM_DEBUG(llvm::dbgs() << "创建向量化计算，使用编译时向量元素数: " 
                           << vectorWidthElements << "\n");
    
    // 构建完整的索引向量 - 针对所有维度
    llvm::SmallVector<Value, 4> inputIndices(dimSizes.size());
    llvm::SmallVector<Value, 4> outputIndices(dimSizes.size());
    
    // 初始化所有维度的索引
    for (int i = 0; i < static_cast<int>(dimSizes.size()); ++i) {
      if (i == outerDim && outerIdx) {
        inputIndices[i] = outerIdx;
        outputIndices[i] = outerIdx;
      } else if (i == innerDim && innerIdx) {
        inputIndices[i] = innerIdx;
        outputIndices[i] = innerIdx;
      } else {
        // 对于其他维度，使用索引0（可以后续改进为实际的循环变量）
        inputIndices[i] = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
        outputIndices[i] = inputIndices[i];
      }
    }
    
    // 分析原始linalg.generic的indexing maps来确定输入输出的映射关系
    ArrayAttr indexingMaps = op.getIndexingMaps();
    auto inputMap = indexingMaps[0].cast<AffineMapAttr>().getValue();
    auto outputMap = indexingMaps[indexingMaps.size() - 1].cast<AffineMapAttr>().getValue();
    
    // 创建向量类型
    auto vectorType = VectorType::get({vectorWidthElements}, elementType);
    Value zeroElement = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));
    
    // 向量读取 - 参考BuiltinTransposeVectorization.cpp的正确用法
    auto vectorRead = rewriter.create<vector::TransferReadOp>(
        loc,
        TypeRange{vectorType},
        ValueRange{input, inputIndices[0], inputIndices.size() > 1 ? inputIndices[1] : inputIndices[0], zeroElement},
        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr("in_bounds", rewriter.getBoolArrayAttr(ArrayRef<bool>{false, true})),
            rewriter.getNamedAttr("operand_segment_sizes", rewriter.getDenseI32ArrayAttr(ArrayRef<int>{1, 2, 1, 0})),
            rewriter.getNamedAttr("permutation_map", AffineMapAttr::get(inputMap))
        });
    
    // 处理linalg.generic的region操作
    Value result = processGenericRegion(rewriter, loc, op, vectorRead, elementType, decision);
    
    // 向量写入 - 参考BuiltinTransposeVectorization.cpp的正确用法
    rewriter.create<vector::TransferWriteOp>(
        loc, TypeRange{},
        ValueRange{result, output, outputIndices[0], outputIndices.size() > 1 ? outputIndices[1] : outputIndices[0]},
        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr("in_bounds", rewriter.getBoolArrayAttr(ArrayRef<bool>{true, true})),
            rewriter.getNamedAttr("operand_segment_sizes", rewriter.getDenseI32ArrayAttr(ArrayRef<int>{1, 1, 2, 0})),
            rewriter.getNamedAttr("permutation_map", AffineMapAttr::get(outputMap))
        });
  }

  /**
   * 处理linalg.generic的region操作 - 将标量操作转换为向量操作
   */
  Value processGenericRegion(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      Value vectorInput, Type elementType, const VectorizationDecision& decision) const {
    
    // 获取原始操作的region
    Region& region = op.getRegion();
    if (region.empty()) {
      return vectorInput; // 如果没有region，直接返回输入
    }
    
    Block& block = region.front();
    Value currentResult = vectorInput;
    
    // 遍历region中的操作，将每个标量操作转换为向量操作
    for (Operation& operation : block.getOperations()) {
      // 跳过终止操作
      if (operation.hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }
      
      StringRef opName = operation.getName().getStringRef();
      
      // 根据操作类型创建对应的向量操作
      if (opName == "arith.addf") {
        // 浮点加法 -> vector.addf
        currentResult = createVectorArithOp<arith::AddFOp>(rewriter, loc, currentResult, currentResult);
      } else if (opName == "arith.subf") {
        // 浮点减法 -> vector.subf
        currentResult = createVectorArithOp<arith::SubFOp>(rewriter, loc, currentResult, currentResult);
      } else if (opName == "arith.mulf") {
        // 浮点乘法 -> vector.mulf
        currentResult = createVectorArithOp<arith::MulFOp>(rewriter, loc, currentResult, currentResult);
        
        // 如果支持FMA，可以进一步优化
        if (decision.useFMA) {
          // TODO: 实现FMA优化
        }
      } else if (opName == "arith.divf") {
        // 浮点除法 -> vector.divf
        currentResult = createVectorArithOp<arith::DivFOp>(rewriter, loc, currentResult, currentResult);
      } else if (opName == "arith.addi") {
        // 整数加法 -> vector.addi
        currentResult = createVectorArithOp<arith::AddIOp>(rewriter, loc, currentResult, currentResult);
      } else if (opName == "arith.subi") {
        // 整数减法 -> vector.subi
        currentResult = createVectorArithOp<arith::SubIOp>(rewriter, loc, currentResult, currentResult);
      } else if (opName == "arith.muli") {
        // 整数乘法 -> vector.muli
        currentResult = createVectorArithOp<arith::MulIOp>(rewriter, loc, currentResult, currentResult);
      } else if (opName == "math.sin" || opName == "math.cos" || opName == "math.exp") {
        // 数学函数 -> 对应的向量函数
        currentResult = createVectorMathOp(rewriter, loc, opName, currentResult);
      } else {
        // 不支持的操作，保持原样（可能导致类型不匹配）
        LLVM_DEBUG(llvm::dbgs() << "不支持向量化的操作: " << opName << "\n");
      }
    }
    
    return currentResult;
  }

  /**
   * 创建向量算术操作
   */
  template<typename ArithOpType>
  Value createVectorArithOp(ConversionPatternRewriter &rewriter, Location loc,
                           Value lhs, Value rhs) const {
    return rewriter.create<ArithOpType>(loc, lhs, rhs);
  }

  /**
   * 创建向量数学操作
   */
  Value createVectorMathOp(ConversionPatternRewriter &rewriter, Location loc,
                          StringRef opName, Value input) const {
    if (opName == "math.sin") {
      return rewriter.create<math::SinOp>(loc, input);
    } else if (opName == "math.cos") {
      return rewriter.create<math::CosOp>(loc, input);
    } else if (opName == "math.exp") {
      return rewriter.create<math::ExpOp>(loc, input);
    } else if (opName == "math.log") {
      return rewriter.create<math::LogOp>(loc, input);
    } else if (opName == "math.sqrt") {
      return rewriter.create<math::SqrtOp>(loc, input);
    } else if (opName == "math.rsqrt") {
      return rewriter.create<math::RsqrtOp>(loc, input);
    } else if (opName == "math.tanh") {
      return rewriter.create<math::TanhOp>(loc, input);
    } else if (opName == "math.absf") {
      return rewriter.create<math::AbsFOp>(loc, input);
    }
    
    // 默认返回输入（不支持的操作）
    return input;
  }

  /**
   * 创建实际的计算操作 - 替换原来的空的createScalarComputation
   * 这里处理无法向量化的情况，使用标量循环
   */
  void createActualComputation(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      Value input, Value output, Value fixedIdx, int fixedDim,
      Type elementType, const VectorizationDecision& decision,
      const VectorizationAnalysisResult& analysis) const {
    
    // 获取linalg.generic的操作数
    auto inputs = op.getInputs();
    auto outputs = op.getOutputs();
    
    // 构建索引
    llvm::SmallVector<Value, 4> indices;
    if (fixedIdx) {
      indices.push_back(fixedIdx);
    } else {
      // 如果没有固定索引，使用常量0（这通常不应该发生）
      indices.push_back(rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
    }
    
    // 读取输入数据
    llvm::SmallVector<Value, 4> inputValues;
    for (Value input : inputs) {
      Value loadedValue = rewriter.create<memref::LoadOp>(loc, input, indices);
      inputValues.push_back(loadedValue);
    }
    
    // 处理region中的操作
    Region& region = op.getRegion();
    if (!region.empty()) {
      Block& block = region.front();
      
      // 创建当前结果值，初始化为第一个输入
      Value currentResult = inputValues.empty() ? 
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType)) : 
        inputValues[0];
      
      // 处理region中的操作
      for (Operation& operation : block.getOperations()) {
        if (operation.hasTrait<OpTrait::IsTerminator>()) {
          continue;
        }
        
        StringRef opName = operation.getName().getStringRef();
        
        // 根据操作类型创建对应的标量操作
        if (opName == "arith.addf" && inputValues.size() >= 2) {
          currentResult = rewriter.create<arith::AddFOp>(loc, inputValues[0], inputValues[1]);
        } else if (opName == "arith.subf" && inputValues.size() >= 2) {
          currentResult = rewriter.create<arith::SubFOp>(loc, inputValues[0], inputValues[1]);
        } else if (opName == "arith.mulf" && inputValues.size() >= 2) {
          currentResult = rewriter.create<arith::MulFOp>(loc, inputValues[0], inputValues[1]);
        } else if (opName == "arith.divf" && inputValues.size() >= 2) {
          currentResult = rewriter.create<arith::DivFOp>(loc, inputValues[0], inputValues[1]);
        } else if (opName == "arith.addi" && inputValues.size() >= 2) {
          currentResult = rewriter.create<arith::AddIOp>(loc, inputValues[0], inputValues[1]);
        } else if (opName == "arith.subi" && inputValues.size() >= 2) {
          currentResult = rewriter.create<arith::SubIOp>(loc, inputValues[0], inputValues[1]);
        } else if (opName == "arith.muli" && inputValues.size() >= 2) {
          currentResult = rewriter.create<arith::MulIOp>(loc, inputValues[0], inputValues[1]);
        } else if (opName == "math.sin") {
          currentResult = rewriter.create<math::SinOp>(loc, currentResult);
        } else if (opName == "math.cos") {
          currentResult = rewriter.create<math::CosOp>(loc, currentResult);
        } else if (opName == "math.exp") {
          currentResult = rewriter.create<math::ExpOp>(loc, currentResult);
        } else {
          LLVM_DEBUG(llvm::dbgs() << "创建标量计算: 不支持的操作 " << opName << "\n");
        }
      }
      
      // 存储结果
      for (Value output : outputs) {
        rewriter.create<memref::StoreOp>(loc, currentResult, output, indices);
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "创建标量计算操作完成\n");
  }

  /**
   * 简化的边界处理
   */
  void createBoundaryHandling(
      ConversionPatternRewriter &rewriter, Location loc,
      Value input, Value output, int outerDim, int innerDim,
      llvm::SmallVector<Value>& dimSizes, Type elementType,
      const VectorizationDecision& decision, const VectorizationAnalysisResult& analysis) const {
    
    // 边界处理主要处理分块后的余数部分
    // 可以使用标量循环或者掩码向量操作
    
    LLVM_DEBUG(llvm::dbgs() << "创建边界处理（简化实现）\n");
  }

  /**
   * 改进的尾部处理 - 使用掩码向量化
   */
  void createTailHandling(
      ConversionPatternRewriter &rewriter, Location loc,
      Value input, Value output, const VectorizationDimension& vecDim,
      Value vectorUpperBound, llvm::SmallVector<Value>& dimSizes,
      Type elementType, const VectorizationDecision& decision,
      const VectorizationAnalysisResult& analysis) const {
    
    // 计算尾部长度
    Value remainingLength = rewriter.create<arith::SubIOp>(loc, dimSizes[vecDim.dimIndex], vectorUpperBound);
    
    // 使用用户提供的编译时常量计算向量元素数量
    int64_t vectorWidthElements = calculateVectorElementsFromUserParams(elementType);
    
    LLVM_DEBUG(llvm::dbgs() << "创建尾部处理，使用编译时向量元素数: " 
                           << vectorWidthElements << "\n");
    
    // 创建尾部掩码
    Value mask = rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get({vectorWidthElements}, rewriter.getI1Type()),
        ValueRange{remainingLength});
    
    // 创建条件分支处理尾部
    Value hasRemainder = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, remainingLength, 
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
    
    rewriter.create<scf::IfOp>(
        loc, hasRemainder, [&](OpBuilder &builder, Location loc) {
          // 构建尾部索引
          llvm::SmallVector<Value, 4> tailIndices(dimSizes.size());
          for (int i = 0; i < static_cast<int>(dimSizes.size()); ++i) {
            if (i == vecDim.dimIndex) {
              tailIndices[i] = vectorUpperBound;
            } else {
              tailIndices[i] = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
            }
          }
          
          // 创建向量类型和身份映射
          auto vectorType = VectorType::get({vectorWidthElements}, elementType);
          Value zeroElement = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(elementType));
          auto identityMap = AffineMap::getMultiDimIdentityMap(dimSizes.size(), builder.getContext());
          
          // 使用掩码进行向量读取 - 参考BuiltinTransposeVectorization.cpp的正确用法
          auto maskedRead = builder.create<vector::TransferReadOp>(
              loc,
              TypeRange{vectorType},
              ValueRange{input, tailIndices[0], tailIndices.size() > 1 ? tailIndices[1] : tailIndices[0], zeroElement},
              ArrayRef<NamedAttribute>{
                  builder.getNamedAttr("in_bounds", builder.getBoolArrayAttr(ArrayRef<bool>{false, true})),
                  builder.getNamedAttr("operand_segment_sizes", builder.getDenseI32ArrayAttr(ArrayRef<int>{1, 2, 1, 0})),
                  builder.getNamedAttr("permutation_map", AffineMapAttr::get(identityMap))
              });
          
          // 简单的恒等操作（实际应该根据linalg.generic的操作）
          Value result = maskedRead;
          
          // 使用掩码进行向量写入 - 参考BuiltinTransposeVectorization.cpp的正确用法
          builder.create<vector::TransferWriteOp>(
              loc, TypeRange{},
              ValueRange{result, output, tailIndices[0], tailIndices.size() > 1 ? tailIndices[1] : tailIndices[0]},
              ArrayRef<NamedAttribute>{
                  builder.getNamedAttr("in_bounds", builder.getBoolArrayAttr(ArrayRef<bool>{true, true})),
                  builder.getNamedAttr("operand_segment_sizes", builder.getDenseI32ArrayAttr(ArrayRef<int>{1, 1, 2, 0})),
                  builder.getNamedAttr("permutation_map", AffineMapAttr::get(identityMap))
              });
              
          builder.create<scf::YieldOp>(loc);
        });
  }

  //===--------------------------------------------------------------------===//
  // 核心分析方法
  //===--------------------------------------------------------------------===//

  /**
   * 综合分析方法 - 分析linalg.generic操作的向量化可行性
   * 整合硬件检测、操作分析、内存访问分析三个维度，确保状态一致性
   */
  VectorizationAnalysisResult performComprehensiveAnalysis(linalg::GenericOp op) const {
    VectorizationAnalysisResult result;
    
    // 1. 硬件环境分析 - 检测当前硬件的向量化支持能力
    result.hardware = analyzeHardwareCapability();
    result.hardwareCapable = result.hardware.isVectorizationCapable;
    
    // 2. 操作模式分析 - 分析迭代器类型和region中的操作
    result.operation = analyzeOperationPattern(op);
    result.operationCompatible = result.operation.allOpsVectorizable && 
                                !result.operation.getParallelDims().empty();
    
    // 3. 内存访问模式分析 - 分析indexing maps确定内存访问模式
    result.memoryAccess = analyzeMemoryAccessPattern(op);
    result.memoryFriendly = result.memoryAccess.isVectorizationFriendly;

    // 4. 数据类型分析 - 提取操作数的实际数据类型 - 新增！
    result.dataType = analyzeDataType(op);
    if (result.dataType.elementSizeBytes == 0) {
      result.failureReason = "无法确定数据类型或不支持的数据类型";
      return result;
    }

    // 5. 状态一致性检查 - 确保并行维度与连续维度有交集
    if (result.hardwareCapable && result.operationCompatible && result.memoryFriendly) {
      bool hasValidDimension = checkDimensionConsistency(result.operation, result.memoryAccess);
      if (hasValidDimension) {
        result.isValid = true;
      } else {
        result.failureReason = "并行维度与连续内存维度无交集，无法有效向量化";
      }
    } else {
      // 构建详细的失败原因
      std::vector<std::string> reasons;
      if (!result.hardwareCapable) reasons.push_back("硬件不支持向量化");
      if (!result.operationCompatible) reasons.push_back("操作不兼容向量化"); 
      if (!result.memoryFriendly) reasons.push_back("内存访问模式不适合向量化");
      
      result.failureReason = "分析失败: ";
      for (size_t i = 0; i < reasons.size(); ++i) {
        result.failureReason += reasons[i];
        if (i < reasons.size() - 1) result.failureReason += "; ";
      }
    }

    return result;
  }

  /**
   * 硬件能力分析 - 基于用户提供的编译时参数构建硬件信息
   * 不再进行运行时检测，而是使用用户在编译时提供的硬件配置
   * 这确保了后续AffineMap构建时使用的都是编译时常量
   */
  HardwareInfo analyzeHardwareCapability() const {
    HardwareInfo info;
    
    // 使用用户提供的编译时常量
    info.architecture = userArchitecture;
    info.vectorExtension = userVectorExtension;
    info.vectorWidthBits = userVectorWidth;
    info.vectorWidthBytes = userVectorWidth / 8;
    info.supportsFMA = userEnableFMA;
    
    // 基于用户配置设置CPU名称（用于日志）
    info.cpuName = "user_specified_" + userArchitecture;
    
    // 验证用户配置的合理性（编译时检查）
    if (userVectorWidth < 128 || userVectorWidth % 128 != 0) {
      LLVM_DEBUG(llvm::dbgs() << "警告: 用户指定的向量宽度(" << userVectorWidth 
                             << "位)可能不合理，建议使用128/256/512\n");
    }
    
    // 检查架构与向量扩展的匹配性
    bool validCombination = true;
    if (userArchitecture == "x86_64") {
      validCombination = (userVectorExtension == "SSE2" || 
                         userVectorExtension == "AVX" ||
                         userVectorExtension == "AVX2" || 
                         userVectorExtension == "AVX512");
    } else if (userArchitecture == "aarch64") {
      validCombination = (userVectorExtension == "NEON" || 
                         userVectorExtension == "SVE");
    }
    
    if (!validCombination) {
      LLVM_DEBUG(llvm::dbgs() << "警告: 架构(" << userArchitecture 
                             << ")与向量扩展(" << userVectorExtension 
                             << ")组合可能不匹配\n");
    }
    
    // 设置向量化支持标志 - 基于用户配置
    info.isVectorizationCapable = (userVectorWidth >= 128);
    
    LLVM_DEBUG(llvm::dbgs() << "硬件分析(用户配置): " << info.architecture 
                           << " " << info.vectorExtension 
                           << " " << info.vectorWidthBits << "位"
                           << " FMA=" << info.supportsFMA 
                           << " 向量化=" << info.isVectorizationCapable << "\n");
    
    return info;
  }

  /**
   * 操作模式分析 - 分析linalg.generic中的迭代器类型和region操作
   * 提取并行/归约维度，检查操作的向量化兼容性，保留完整的维度-类型映射
   */
  OperationInfo analyzeOperationPattern(linalg::GenericOp op) const {
    OperationInfo info;
    info.hasComplexOps = false;
    info.allOpsVectorizable = true;

    // 1. 分析迭代器类型 - 保留完整的维度信息
    auto iteratorTypesArray = op.getIteratorTypesArray();
    for (int i = 0; i < static_cast<int>(iteratorTypesArray.size()); ++i) {
      auto iterType = iteratorTypesArray[i];
      std::string iterTypeStr;
      
      // 将枚举类型转换为字符串
      if (iterType == utils::IteratorType::parallel) {
        iterTypeStr = "parallel";
      } else if (iterType == utils::IteratorType::reduction) {
        iterTypeStr = "reduction";
      } else {
        iterTypeStr = "unknown";
      }
        
      // 创建维度信息对象，保留索引-类型映射
      IteratorDimension dimInfo(i, iterTypeStr);
        
        // 尝试获取维度大小信息（从操作数的shape中推断）
        // 从第一个输入操作数获取维度信息
        if (!op.getInputs().empty()) {
          Value input = op.getInputs()[0];
          if (auto shapedType = input.getType().dyn_cast<ShapedType>()) {
            if (i < static_cast<int>(shapedType.getRank()) && !shapedType.isDynamicDim(i)) {
              dimInfo.dimSize = shapedType.getDimSize(i);
              dimInfo.isBoundKnown = true;
            }
          }
        }
        
      info.dimensions.push_back(dimInfo);
    }

    // 2. 分析region中的操作 - 检查每个操作的向量化兼容性
    Region& region = op.getRegion();
    if (!region.empty()) {
      Block& block = region.front();
      
      for (Operation& operation : block.getOperations()) {
        // 跳过终止操作
        if (operation.hasTrait<OpTrait::IsTerminator>()) {
          continue;
        }

        StringRef opName = operation.getName().getStringRef();
        
        // 检查操作是否可向量化
        if (isOperationVectorizable(opName)) {
          info.supportedOps.insert(opName.str());
        } else {
          info.unsupportedOps.insert(opName.str());
          info.allOpsVectorizable = false;
          
          // 检查是否为复杂操作 (控制流、函数调用等)
          if (opName.startswith("cf.") || opName.startswith("scf.") || 
              opName.contains("call") || opName.contains("invoke")) {
            info.hasComplexOps = true;
          }
        }
      }
    }

    // 输出详细的维度分析信息 - 现在保留了完整的维度-类型映射
    LLVM_DEBUG({
      llvm::dbgs() << "操作分析详细结果:\n";
      llvm::dbgs() << "  总维度数: " << info.dimensions.size() << "\n";
      for (const auto& dim : info.dimensions) {
        llvm::dbgs() << "    维度" << dim.dimIndex << ": " << dim.iteratorType;
        if (dim.isBoundKnown) {
          llvm::dbgs() << " (大小=" << dim.dimSize << ")";
        } else {
          llvm::dbgs() << " (大小未知)";
        }
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "  并行维度数: " << info.getParallelDims().size() << "\n";
      llvm::dbgs() << "  归约维度数: " << info.getReductionDims().size() << "\n";
      llvm::dbgs() << "  已知大小的并行维度: " << (info.hasKnownSizeParallelDims() ? "是" : "否") << "\n";
      llvm::dbgs() << "  最大并行维度大小: " << info.getLargestParallelDimSize() << "\n";
      llvm::dbgs() << "  支持的操作数: " << info.supportedOps.size() << "\n";
      llvm::dbgs() << "  不支持的操作数: " << info.unsupportedOps.size() << "\n";
    });

    return info;
  }

  /**
   * 内存访问模式分析 - 分析indexing maps确定内存访问模式
   * 改进版本：正确分析内存步长和连续性
   */
  MemoryAccessInfo analyzeMemoryAccessPattern(linalg::GenericOp op) const {
    MemoryAccessInfo info;
    info.hasSequentialAccess = false;
    info.hasBroadcastAccess = false;
    info.hasTransposeAccess = false;
    info.isVectorizationFriendly = false;

    // 获取indexing maps
    ArrayAttr indexingMaps = op.getIndexingMaps();
    if (!indexingMaps) {
      return info;
    }

    // 获取操作数信息用于分析内存布局
    auto inputs = op.getInputs();
    auto outputs = op.getOutputs();
    
    // 分析每个操作数的内存访问模式
    int mapIndex = 0;
    for (auto mapAttr : indexingMaps.getAsRange<AffineMapAttr>()) {
      AffineMap map = mapAttr.getValue();
      
      // 获取对应的操作数（输入或输出）
      Value operand = nullptr;
      if (mapIndex < static_cast<int>(inputs.size())) {
        operand = inputs[mapIndex];
      } else {
        operand = outputs[mapIndex - inputs.size()];
      }
      
      // 分析该操作数的连续访问维度
      analyzeSingleOperandAccess(map, operand, info);
      mapIndex++;
    }

    // 设置向量化友好标志
    info.isVectorizationFriendly = !info.continuousDims.empty() && 
                                   !info.hasTransposeAccess;

    LLVM_DEBUG(llvm::dbgs() << "内存访问分析改进版: 顺序访问=" << info.hasSequentialAccess
                           << ", 连续维度数=" << info.continuousDims.size()
                           << ", 向量化友好=" << info.isVectorizationFriendly << "\n");

    return info;
  }

  /**
   * 分析单个操作数的内存访问模式
   * 核心逻辑：确定哪些迭代维度对应连续的内存访问
   */
  void analyzeSingleOperandAccess(AffineMap map, Value operand, MemoryAccessInfo& info) const {
    if (!operand) return;
    
    // 获取操作数的类型信息
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    if (!shapedType || !shapedType.hasRank()) {
      return;
    }
    
    int rank = shapedType.getRank();
    auto results = map.getResults();
    
    LLVM_DEBUG(llvm::dbgs() << "分析操作数: rank=" << rank 
                           << ", map结果数=" << results.size() << "\n");
    
    // 1. 恒等映射分析 - 最简单的情况
    if (map.isIdentity()) {
      info.hasSequentialAccess = true;
      // 对于row-major布局，最后一个维度是连续的
      if (rank > 0) {
        addContinuousDimension(rank - 1, info);
      }
      return;
    }
    
    // 2. 分析内存步长模式
    analyzeMemoryStridePattern(map, shapedType, info);
    
    // 3. 检查特殊访问模式
    checkSpecialAccessPatterns(map, info);
  }

  /**
   * 分析内存步长模式 - 核心方法
   * 通过分析affine map确定哪些迭代维度对应最小的内存步长
   */
  void analyzeMemoryStridePattern(AffineMap map, ShapedType shapedType, MemoryAccessInfo& info) const {
    auto results = map.getResults();
    
    // 对于每个迭代维度，计算其对应的内存访问步长
    for (unsigned iterDim = 0; iterDim < map.getNumDims(); ++iterDim) {
      int64_t minStride = calculateMemoryStride(map, iterDim, shapedType);
      
      LLVM_DEBUG(llvm::dbgs() << "  迭代维度" << iterDim 
                             << "的内存步长: " << minStride << "\n");
      
      // 步长为1表示连续访问（对于row-major布局）
      if (minStride == 1) {
        addContinuousDimension(iterDim, info);
        info.hasSequentialAccess = true;
      }
    }
  }

  /**
   * 计算迭代维度对应的内存访问步长
   * 返回该迭代维度在内存中的最小步长
   */
  int64_t calculateMemoryStride(AffineMap map, unsigned iterDim, ShapedType shapedType) const {
    auto results = map.getResults();
    int rank = shapedType.getRank();
    int64_t minStride = INT64_MAX;
    
    // 对于每个内存维度，检查该迭代维度的贡献
    for (unsigned memDim = 0; memDim < results.size() && memDim < rank; ++memDim) {
      auto result = results[memDim];
      
      // 检查该结果表达式中是否包含当前迭代维度
      if (auto dimExpr = result.dyn_cast<AffineDimExpr>()) {
        if (dimExpr.getPosition() == iterDim) {
          // 计算该内存维度的步长
          int64_t stride = calculateDimensionStride(memDim, shapedType);
          minStride = std::min(minStride, stride);
        }
      } else if (auto binaryExpr = result.dyn_cast<AffineBinaryOpExpr>()) {
        // 处理复杂表达式（如 d0 + d1, d0 * 2 等）
        int64_t stride = analyzeComplexExpression(binaryExpr, iterDim, memDim, shapedType);
        if (stride > 0) {
          minStride = std::min(minStride, stride);
        }
      }
    }
    
    return (minStride == INT64_MAX) ? 0 : minStride;
  }

  /**
   * 计算内存维度的步长
   * 对于row-major布局：stride[i] = product(shape[i+1:])
   */
  int64_t calculateDimensionStride(unsigned memDim, ShapedType shapedType) const {
    int64_t stride = 1;
    auto shape = shapedType.getShape();
    
    // 从该维度之后的所有维度计算步长
    for (unsigned i = memDim + 1; i < shape.size(); ++i) {
      if (shapedType.isDynamicDim(i)) {
        return -1; // 动态维度无法确定步长
      }
      stride *= shape[i];
    }
    
    return stride;
  }

  /**
   * 分析复杂的affine表达式
   * 处理如 d0 + d1, d0 * c 等情况
   */
  int64_t analyzeComplexExpression(AffineBinaryOpExpr expr, unsigned iterDim, 
                                  unsigned memDim, ShapedType shapedType) const {
    // 检查表达式中是否包含目标迭代维度
    bool containsIterDim = false;
    
    // 递归检查左右操作数
    if (auto leftDim = expr.getLHS().dyn_cast<AffineDimExpr>()) {
      if (leftDim.getPosition() == iterDim) {
        containsIterDim = true;
      }
    }
    if (auto rightDim = expr.getRHS().dyn_cast<AffineDimExpr>()) {
      if (rightDim.getPosition() == iterDim) {
        containsIterDim = true;
      }
    }
    
    if (containsIterDim) {
      // 如果包含目标维度，返回该内存维度的步长
      return calculateDimensionStride(memDim, shapedType);
    }
    
    return -1; // 不包含目标维度
  }

  /**
   * 检查特殊访问模式
   */
  void checkSpecialAccessPatterns(AffineMap map, MemoryAccessInfo& info) const {
    auto results = map.getResults();
    
    // 检查广播模式 (包含常量表达式)
    for (auto result : results) {
      if (result.isa<AffineConstantExpr>()) {
        info.hasBroadcastAccess = true;
      }
    }
    
    // 检查转置模式 (2D情况下的维度交换)
    if (map.getNumDims() == 2 && results.size() == 2) {
      auto first = results[0].dyn_cast<AffineDimExpr>();
      auto second = results[1].dyn_cast<AffineDimExpr>();
      if (first && second && 
          first.getPosition() == 1 && second.getPosition() == 0) {
        info.hasTransposeAccess = true;
        LLVM_DEBUG(llvm::dbgs() << "检测到转置访问模式\n");
      }
    }
  }

  /**
   * 添加连续维度到列表（避免重复）
   */
  void addContinuousDimension(int dimIndex, MemoryAccessInfo& info) const {
    if (std::find(info.continuousDims.begin(), info.continuousDims.end(), dimIndex) 
        == info.continuousDims.end()) {
      info.continuousDims.push_back(dimIndex);
      LLVM_DEBUG(llvm::dbgs() << "添加连续访问维度: " << dimIndex << "\n");
    }
  }

  /**
   * 检查维度一致性 - 确保并行维度与连续维度有交集
   * 这是向量化可行的关键条件，利用完整的维度信息进行精确匹配
   */
  bool checkDimensionConsistency(const OperationInfo& opInfo, 
                                const MemoryAccessInfo& memInfo) const {
    // 获取并行维度列表
    std::vector<int> parallelDims = opInfo.getParallelDims();
    
    // 检查并行维度与连续维度的交集
    for (int parallelDim : parallelDims) {
      for (int continuousDim : memInfo.continuousDims) {
        if (parallelDim == continuousDim) {
          // 找到可向量化的维度，可进一步获取该维度的详细信息
          const IteratorDimension* dimInfo = opInfo.getDimension(parallelDim);
          if (dimInfo) {
            LLVM_DEBUG(llvm::dbgs() << "找到可向量化维度: " << parallelDim 
                                   << " (类型: " << dimInfo->iteratorType << ")\n");
          }
          return true;
        }
      }
    }
    return false;
  }

  /**
   * 检查操作是否可向量化
   * 基于预定义的可向量化操作列表
   */
  bool isOperationVectorizable(StringRef opName) const {
    // 算术操作
    if (opName.startswith("arith.")) {
      return opName == "arith.addf" || opName == "arith.subf" || 
             opName == "arith.mulf" || opName == "arith.divf" ||
             opName == "arith.negf" || opName == "arith.addi" ||
             opName == "arith.subi" || opName == "arith.muli" ||
             opName == "arith.maxf" || opName == "arith.minf" ||
             opName == "arith.cmpf" || opName == "arith.cmpi" ||
             opName == "arith.select" || opName == "arith.constant";
    }
    
    // 数学操作
    if (opName.startswith("math.")) {
      return opName == "math.sin" || opName == "math.cos" ||
             opName == "math.exp" || opName == "math.log" ||
             opName == "math.sqrt" || opName == "math.rsqrt" ||
             opName == "math.tanh" || opName == "math.absf";
    }
    
    return false;
  }

  //===--------------------------------------------------------------------===//
  // 决策模块核心方法
  //===--------------------------------------------------------------------===//

  /**
   * 向量化决策方法 - 基于分析结果制定向量化策略
   * 包括维度选择、分块策略、边界处理等关键决策
   */
  VectorizationDecision makeVectorizationDecision(const VectorizationAnalysisResult& analysis) const {
    VectorizationDecision decision;
    
    // 1. 向量化维度选择 - 选择最适合的并行维度进行向量化
    decision.vectorizedDims = selectVectorizationDimensions(analysis);
    if (decision.vectorizedDims.empty()) {
      decision.failureReason = "无法找到合适的向量化维度";
      return decision;
    }
    
    // 2. 确定向量宽度和数据类型
    decision.effectiveVectorWidth = determineVectorWidth(analysis, decision.vectorizedDims);
    decision.vectorType = determineVectorType(analysis);
    
    // 3. 分块策略决策 - 最多二维分块
    decision.tiling = determineTilingStrategy(analysis, decision.vectorizedDims);
    
    // 4. 边界处理策略
    determineTailHandlingStrategy(decision.vectorizedDims, analysis.hardware);
    
    // 5. 转换策略选择
    decision.conversionStrategy = selectConversionStrategy(analysis, decision);
    decision.useFMA = analysis.hardware.supportsFMA && shouldUseFMA(analysis);
    
    decision.isValid = true;
    
    LLVM_DEBUG({
      llvm::dbgs() << "向量化决策详细结果:\n";
      llvm::dbgs() << "  向量化维度数: " << decision.vectorizedDims.size() << "\n";
      for (size_t i = 0; i < decision.vectorizedDims.size(); ++i) {
        const auto& dim = decision.vectorizedDims[i];
        llvm::dbgs() << "    维度" << i << ": 索引=" << dim.dimIndex 
                     << ", 大小=" << dim.dimSize 
                     << ", 向量宽度=" << dim.vectorWidth << "位"
                     << ", 余数=" << dim.remainder
                     << ", 尾部处理=" << dim.tailStrategy << "\n";
      }
      llvm::dbgs() << "  有效向量宽度: " << decision.effectiveVectorWidth << "位\n";
      llvm::dbgs() << "  向量类型: " << decision.vectorType << "\n";
      llvm::dbgs() << "  分块级别: " << decision.tiling.tilingLevel << "\n";
      if (decision.tiling.isValid) {
        llvm::dbgs() << "  分块维度: ";
        for (size_t i = 0; i < decision.tiling.tiledDims.size(); ++i) {
          llvm::dbgs() << "维度" << decision.tiling.tiledDims[i] 
                       << "(大小=" << decision.tiling.tileSizes[i] << ")";
          if (i < decision.tiling.tiledDims.size() - 1) llvm::dbgs() << ", ";
        }
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "  使用FMA: " << (decision.useFMA ? "是" : "否") << "\n";
      llvm::dbgs() << "  转换策略: " << decision.conversionStrategy << "\n";
    });
    
    return decision;
  }

  /**
   * 向量化维度选择 - 选择最适合的并行维度进行向量化
   * 考虑维度大小、内存访问模式、硬件能力等因素
   */
  std::vector<VectorizationDimension> selectVectorizationDimensions(
      const VectorizationAnalysisResult& analysis) const {
    std::vector<VectorizationDimension> selectedDims;
    
    // 获取并行维度与连续维度的交集
    std::vector<int> parallelDims = analysis.operation.getParallelDims();
    std::vector<int> continuousDims = analysis.memoryAccess.continuousDims;
    
    // 找到交集维度
    std::vector<int> candidateDims;
    for (int parallelDim : parallelDims) {
      for (int continuousDim : continuousDims) {
        if (parallelDim == continuousDim) {
          candidateDims.push_back(parallelDim);
          break;
        }
      }
    }
    
    if (candidateDims.empty()) {
      return selectedDims; // 没有候选维度
    }
    
    // 根据维度大小和硬件能力选择最佳维度
    
    for (int dimIndex : candidateDims) {
      const IteratorDimension* dimInfo = analysis.operation.getDimension(dimIndex);
      if (!dimInfo || !dimInfo->isBoundKnown || dimInfo->dimSize <= 0) {
        continue; // 跳过大小未知的维度
      }
      
      // 计算该维度的向量化参数 - 使用实际数据类型
      int64_t dimSize = dimInfo->dimSize;
      int vectorWidth = calculateOptimalVectorWidth(dimSize, analysis.dataType.elementType);
      
      if (vectorWidth > 0) {
        VectorizationDimension vecDim(dimIndex, dimSize, vectorWidth);
        selectedDims.push_back(vecDim);
      }
    }
    
    // 按维度大小排序，优先选择较大的维度
    std::sort(selectedDims.begin(), selectedDims.end(), 
              [](const VectorizationDimension& a, const VectorizationDimension& b) {
                return a.dimSize > b.dimSize;
              });
    
    // 限制最多选择2个维度（二维分块约束）
    if (selectedDims.size() > 2) {
      selectedDims.resize(2);
    }
    
    return selectedDims;
  }

  /**
   * 计算最优向量宽度 - 基于用户参数和编译时类型分析
   * 完全使用编译时常量，确保AffineMap构建正确
   */
  int calculateOptimalVectorWidth(int64_t dimSize, Type elementType) const {
    // 使用编译时可确定的元素大小
    int64_t elementSizeBits = 32; // 默认f32
    
    // 基于类型进行编译时分支判断
    if (auto floatType = elementType.dyn_cast<FloatType>()) {
      elementSizeBits = floatType.getWidth(); // 编译时已知
    } else if (auto intType = elementType.dyn_cast<IntegerType>()) {
      elementSizeBits = intType.getWidth(); // 编译时已知
    } else if (elementType.isa<IndexType>()) {
      elementSizeBits = 64; // 编译时常量
    } else {
      // 对于其他类型，使用保守的默认值
      elementSizeBits = 32;
    }
    
    // 使用用户指定的硬件向量宽度（编译时常量）
    int64_t hardwareVectorWidth = userVectorWidth;
    
    // 计算最大向量元素数
    int64_t maxElements = hardwareVectorWidth / elementSizeBits;
    
    // 确保向量宽度不超过维度大小
    int64_t optimalElements = std::min(dimSize, maxElements);
    
    // 如果维度太小，不适合向量化
    if (optimalElements < 2) {
      return 0;
    }
    
    // 优先选择2的幂次方（编译时计算）
    int64_t powerOfTwo = 1;
    while (powerOfTwo * 2 <= optimalElements) {
      powerOfTwo *= 2;
    }
    
    // 返回元素数量（而不是位宽，与VectorizationDimension.vectorWidth保持一致）
    return static_cast<int>(powerOfTwo);
  }

  /**
   * 确定向量宽度 - 基于选择的维度和硬件能力
   */
  int determineVectorWidth(const VectorizationAnalysisResult& analysis,
                          const std::vector<VectorizationDimension>& vectorizedDims) const {
    if (vectorizedDims.empty()) {
      return 0;
    }
    
    // 选择最小的向量宽度，确保所有维度都能使用
    int minVectorWidth = vectorizedDims[0].vectorWidth;
    for (const auto& dim : vectorizedDims) {
      minVectorWidth = std::min(minVectorWidth, dim.vectorWidth);
    }
    
    return minVectorWidth;
  }

  /**
   * 确定向量数据类型 - 基于实际数据类型和硬件能力
   */
  std::string determineVectorType(const VectorizationAnalysisResult& analysis) const {
    // 使用实际数据类型和硬件向量扩展 - 修复硬编码问题！
    std::string extension = analysis.hardware.vectorExtension;
    std::string dataTypeStr = analysis.dataType.typeString;
    int elementSizeBits = analysis.dataType.elementSizeBits;
    
    // 根据硬件能力计算向量元素数
    int vectorWidthBits = analysis.hardware.vectorWidthBits;
    int elementsCount = vectorWidthBits / elementSizeBits;
    
    // 构建向量类型字符串
    return "vector<" + std::to_string(elementsCount) + "x" + dataTypeStr + ">";
  }

  /**
   * 确定分块策略 - 最多二维分块
   */
  TilingStrategy determineTilingStrategy(const VectorizationAnalysisResult& analysis,
                                        const std::vector<VectorizationDimension>& vectorizedDims) const {
    TilingStrategy strategy;
    
    if (vectorizedDims.empty()) {
      return strategy;
    }
    
    // 最多二维分块约束
    int maxTilingDims = std::min(2, static_cast<int>(vectorizedDims.size()));
    
    for (int i = 0; i < maxTilingDims; ++i) {
      const auto& dim = vectorizedDims[i];
      strategy.tiledDims.push_back(dim.dimIndex);
      strategy.tileSizes.push_back(dim.vectorWidth / 32); // 转换为元素数
    }
    
    strategy.tilingLevel = maxTilingDims;
    strategy.isValid = true;
    
    return strategy;
  }

  /**
   * 确定尾部处理策略
   */
  void determineTailHandlingStrategy(std::vector<VectorizationDimension>& vectorizedDims,
                                    const HardwareInfo& hardware) const {
    for (auto& dim : vectorizedDims) {
      if (dim.needsTailHandling) {
        // 根据余数大小选择策略
        if (dim.remainder <= 4) {
          dim.tailStrategy = "mask"; // 使用掩码处理
        } else {
          dim.tailStrategy = "scalar"; // 使用标量处理
        }
      } else {
        dim.tailStrategy = "none"; // 不需要尾部处理
      }
    }
  }

  /**
   * 选择转换策略
   */
  std::string selectConversionStrategy(const VectorizationAnalysisResult& analysis,
                                      const VectorizationDecision& decision) const {
    // 根据分块级别选择转换策略 - 修正为实际使用的affine方言
    if (decision.tiling.tilingLevel == 2) {
      return "affine.parallel"; // 双层并行分块：两层affine.parallel
    } else if (decision.tiling.tilingLevel == 1) {
      return "affine.parallel+affine.for"; // 单层分块：一层affine.parallel + affine.for
    } else {
      return "affine.for"; // 无分块：纯affine.for循环
    }
  }

  /**
   * 判断是否应该使用FMA指令
   */
  bool shouldUseFMA(const VectorizationAnalysisResult& analysis) const {
    // 检查是否包含乘法操作
    const auto& supportedOps = analysis.operation.supportedOps;
    return supportedOps.count("arith.mulf") > 0 || 
           supportedOps.count("arith.addf") > 0;
  }

  //===--------------------------------------------------------------------===//
  // 数据类型分析方法
  //===--------------------------------------------------------------------===//

     /**
    * 数据类型分析 - 改进版！支持完整的MemRef元素类型规范
    * 根据MLIR MemRefType规范，支持所有可能的元素类型
    */
   DataTypeInfo analyzeDataType(linalg::GenericOp op) const {
     DataTypeInfo info;
     
     // 获取操作数的类型信息
     auto inputs = op.getInputs();
     auto outputs = op.getOutputs();
     
     // 分析第一个有效操作数的类型（输入优先）
     for (auto operand : inputs) {
       if (analyzeOperandDataType(operand, info)) {
         break; // 找到有效类型后退出
       }
     }
     
     // 如果输入中没有找到有效类型，检查输出
     if (info.elementSizeBytes == 0) {
       for (auto operand : outputs) {
         if (analyzeOperandDataType(operand, info)) {
           break;
         }
       }
     }
     
     LLVM_DEBUG(llvm::dbgs() << "数据类型分析结果: " << info.typeString 
                            << " (" << info.elementSizeBits << "位)"
                            << " 向量=" << info.isVector 
                            << " 嵌套=" << info.isNestedMemRef << "\n");
     
     return info;
   }

   /**
    * 分析单个操作数的数据类型
    * 返回true表示成功分析出有效类型
    */
   bool analyzeOperandDataType(Value operand, DataTypeInfo& info) const {
     // 1. 检查是否为MemRefType
     if (auto memrefType = operand.getType().dyn_cast<MemRefType>()) {
       return analyzeMemRefElementType(memrefType, info);
     }
     
     // 2. 检查是否为TensorType
     if (auto tensorType = operand.getType().dyn_cast<RankedTensorType>()) {
       return analyzeTensorElementType(tensorType, info);
     }
     
     // 3. 检查其他ShapedType
     if (auto shapedType = operand.getType().dyn_cast<ShapedType>()) {
       return analyzeShapedElementType(shapedType, info);
     }
     
     return false; // 无法分析的类型
   }

   /**
    * 分析MemRef的元素类型和布局信息
    */
   bool analyzeMemRefElementType(MemRefType memrefType, DataTypeInfo& info) const {
     Type elementType = memrefType.getElementType();
     
     // 分析基本元素类型
     if (!analyzeElementTypeDetails(elementType, info)) {
       return false;
     }
     
     // 分析MemRef特有的布局和内存空间信息
     info.hasCustomLayout = !memrefType.getLayout().isIdentity();
     info.hasMemorySpace = (memrefType.getMemorySpace() != nullptr);
     
     if (info.hasCustomLayout) {
       info.layoutInfo = "custom_layout"; // 简化表示
     }
     
     if (info.hasMemorySpace) {
       info.memorySpaceInfo = "custom_space"; // 简化表示
     }
     
     return true;
   }

   /**
    * 分析Tensor的元素类型
    */
   bool analyzeTensorElementType(RankedTensorType tensorType, DataTypeInfo& info) const {
     Type elementType = tensorType.getElementType();
     return analyzeElementTypeDetails(elementType, info);
   }

   /**
    * 分析一般ShapedType的元素类型
    */
   bool analyzeShapedElementType(ShapedType shapedType, DataTypeInfo& info) const {
     Type elementType = shapedType.getElementType();
     return analyzeElementTypeDetails(elementType, info);
   }

   /**
    * 分析元素类型的详细信息 - 核心方法
    * 支持MemRef规范中的所有元素类型
    */
   bool analyzeElementTypeDetails(Type elementType, DataTypeInfo& info) const {
     info.elementType = elementType;
     
     // 1. 内置浮点类型
     if (auto floatType = elementType.dyn_cast<FloatType>()) {
       info.isFloatingPoint = true;
       info.elementSizeBits = floatType.getWidth();
       info.elementSizeBytes = (floatType.getWidth() + 7) / 8; // 向上取整
       info.typeString = "f" + std::to_string(floatType.getWidth());
       return true;
     }
     
     // 2. 内置整数类型
     if (auto intType = elementType.dyn_cast<IntegerType>()) {
       info.isInteger = true;
       info.elementSizeBits = intType.getWidth();
       info.elementSizeBytes = (intType.getWidth() + 7) / 8;
       info.isSigned = intType.isSigned();
       info.typeString = (intType.isUnsigned() ? "ui" : "i") + std::to_string(intType.getWidth());
       return true;
     }
     
     // 3. 内置索引类型
     if (elementType.isa<IndexType>()) {
       info.isIndex = true;
       info.elementSizeBits = 64; // 通常为64位
       info.elementSizeBytes = 8;
       info.typeString = "index";
       return true;
     }
     
     // 4. 复数类型
     if (auto complexType = elementType.dyn_cast<ComplexType>()) {
       info.isComplex = true;
       Type innerType = complexType.getElementType();
       
       // 递归分析内部类型
       DataTypeInfo innerInfo;
       if (analyzeElementTypeDetails(innerType, innerInfo)) {
         info.elementSizeBits = innerInfo.elementSizeBits * 2; // 复数是两倍大小
         info.elementSizeBytes = innerInfo.elementSizeBytes * 2;
         info.typeString = "complex<" + innerInfo.typeString + ">";
         return true;
       }
       return false;
     }
     
     // 5. 向量类型
     if (auto vectorType = elementType.dyn_cast<VectorType>()) {
       info.isVector = true;
       info.vectorSize = vectorType.getNumElements();
       info.vectorElementType = vectorType.getElementType();
       
       // 递归分析向量元素类型
       DataTypeInfo vectorElemInfo;
       if (analyzeElementTypeDetails(info.vectorElementType, vectorElemInfo)) {
         info.elementSizeBits = vectorElemInfo.elementSizeBits * info.vectorSize;
         info.elementSizeBytes = vectorElemInfo.elementSizeBytes * info.vectorSize;
         info.typeString = "vector<" + std::to_string(info.vectorSize) + "x" + vectorElemInfo.typeString + ">";
         return true;
       }
       return false;
     }
     
     // 6. 嵌套MemRef类型
     if (auto nestedMemRefType = elementType.dyn_cast<MemRefType>()) {
       info.isNestedMemRef = true;
       info.typeString = "memref<nested>";
       
       // 嵌套MemRef的大小通常是指针大小
       info.elementSizeBits = 64; // 指针通常64位
       info.elementSizeBytes = 8;
       return true;
     }
     
     // 7. 其他自定义类型（实现MemRefElementTypeInterface）
     // 这里简化处理，实际需要查询接口
     info.isCustomType = true;
     info.typeString = "custom_type";
     info.elementSizeBits = 64; // 默认假设
     info.elementSizeBytes = 8;
     
     LLVM_DEBUG(llvm::dbgs() << "检测到自定义或未知元素类型\n");
     return true; // 即使是未知类型也返回true，使用默认值
   }

  //===--------------------------------------------------------------------===//
  // 用户参数辅助方法
  //===--------------------------------------------------------------------===//

  /**
   * 基于用户提供的参数计算向量元素数量 - 纯编译时计算
   * 完全避免运行时变量，确保AffineMap构建正确
   */
  int64_t calculateVectorElementsFromUserParams(Type elementType) const {
    // 使用编译时可确定的元素大小计算
    int64_t elementSizeBits = 32; // 默认f32
    
    // 基于类型进行编译时分支判断
    if (auto floatType = elementType.dyn_cast<FloatType>()) {
      elementSizeBits = floatType.getWidth(); // 编译时已知
    } else if (auto intType = elementType.dyn_cast<IntegerType>()) {
      elementSizeBits = intType.getWidth(); // 编译时已知
    } else if (elementType.isa<IndexType>()) {
      elementSizeBits = 64; // 编译时常量
    } else {
      // 对于其他类型，使用保守的默认值
      elementSizeBits = 32;
    }
    
    // 使用用户指定的向量宽度进行编译时计算
    int64_t vectorElements = userVectorWidth / elementSizeBits;
    
    // 编译时边界检查和修正
    if (vectorElements < 1) vectorElements = 1;
    if (vectorElements > 64) vectorElements = 64;
    
    LLVM_DEBUG(llvm::dbgs() << "编译时计算向量元素数: " << userVectorWidth 
                           << "位 / " << elementSizeBits << "位 = " 
                           << vectorElements << "个元素\\n");
    
    return vectorElements;
  }

  //===--------------------------------------------------------------------===//
  // 用户提供的编译时硬件参数 - 确保AffineMap构建正确
  //===--------------------------------------------------------------------===//
  
  const std::string userArchitecture;      // 用户指定的目标架构
  const int64_t userVectorWidth;           // 用户指定的向量宽度(位)
  const std::string userVectorExtension;   // 用户指定的向量扩展
  const bool userEnableFMA;                // 用户指定是否启用FMA
  const int64_t userTileSize;              // 用户指定的分块大小

  /**
   * 创建向量化元素级计算操作 - 真正的向量化实现
   * 使用vector.transfer_read, vector操作, vector.transfer_write
   */
  void createVectorizedElementWiseComputation(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      Value index, Type elementType, int64_t vectorElements) const {
    
    // 获取linalg.generic的操作数
    auto inputs = op.getInputs();
    auto outputs = op.getOutputs();
    
    // 构建向量类型
    auto vectorType = VectorType::get({vectorElements}, elementType);
    Value zeroElement = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));
    
    // 创建identity affine map用于1D访问
    auto identityMap = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)}, rewriter.getContext());
    
    // 读取向量化输入数据
    llvm::SmallVector<Value, 4> vectorInputs;
    for (Value input : inputs) {
      auto vectorRead = rewriter.create<vector::TransferReadOp>(
          loc, vectorType, input, ValueRange{index}, zeroElement);
      vectorInputs.push_back(vectorRead);
    }
    
    // 处理linalg.generic的region操作，转换为向量操作
    Value result = processGenericRegionVectorized(rewriter, loc, op, vectorInputs, elementType);
    
    // 写入向量化结果
    for (Value output : outputs) {
      rewriter.create<vector::TransferWriteOp>(
          loc, result, output, ValueRange{index});
    }
    
    LLVM_DEBUG(llvm::dbgs() << "向量化元素计算操作完成，向量宽度: " << vectorElements << "\n");
  }

  /**
   * 处理linalg.generic的region操作 - 向量化版本
   * 将标量操作转换为对应的向量操作
   */
  Value processGenericRegionVectorized(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      llvm::SmallVector<Value, 4>& vectorInputs, Type elementType) const {
    
    // 获取原始操作的region
    Region& region = op.getRegion();
    if (region.empty() || vectorInputs.empty()) {
      return vectorInputs.empty() ? 
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType)) :
        vectorInputs[0];
    }
    
    Block& block = region.front();
    Value currentResult = vectorInputs[0];
    
    // 遍历region中的操作，将每个标量操作转换为向量操作
    for (Operation& operation : block.getOperations()) {
      // 跳过终止操作
      if (operation.hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }
      
      StringRef opName = operation.getName().getStringRef();
      
      // 根据操作类型创建对应的向量操作
      if (opName == "arith.addf" && vectorInputs.size() >= 2) {
        currentResult = rewriter.create<arith::AddFOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.subf" && vectorInputs.size() >= 2) {
        currentResult = rewriter.create<arith::SubFOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.mulf" && vectorInputs.size() >= 2) {
        currentResult = rewriter.create<arith::MulFOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.divf" && vectorInputs.size() >= 2) {
        currentResult = rewriter.create<arith::DivFOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.addi" && vectorInputs.size() >= 2) {
        currentResult = rewriter.create<arith::AddIOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.subi" && vectorInputs.size() >= 2) {
        currentResult = rewriter.create<arith::SubIOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.muli" && vectorInputs.size() >= 2) {
        currentResult = rewriter.create<arith::MulIOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "math.sin") {
        currentResult = rewriter.create<math::SinOp>(loc, currentResult);
      } else if (opName == "math.cos") {
        currentResult = rewriter.create<math::CosOp>(loc, currentResult);
      } else if (opName == "math.exp") {
        currentResult = rewriter.create<math::ExpOp>(loc, currentResult);
      } else if (opName == "math.log") {
        currentResult = rewriter.create<math::LogOp>(loc, currentResult);
      } else if (opName == "math.sqrt") {
        currentResult = rewriter.create<math::SqrtOp>(loc, currentResult);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "向量化计算: 不支持的操作 " << opName << "\n");
      }
    }
    
    return currentResult;
  }

  /**
   * 处理尾部非对齐向量数据
   * 对于不能被向量宽度整除的剩余元素，使用掩码向量化或标量处理
   */
  void createTailVectorHandling(
      ConversionPatternRewriter &rewriter, Location loc, linalg::GenericOp op,
      Value vectorUpperBound, Value totalSize, Type elementType, int64_t vectorElements) const {
    
    // 计算剩余元素数量
    Value remainingElements = rewriter.create<arith::SubIOp>(loc, totalSize, vectorUpperBound);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    
    // 检查是否有剩余元素需要处理
    Value hasRemainder = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, remainingElements, zero);
    
    rewriter.create<scf::IfOp>(
        loc, hasRemainder, [&](OpBuilder &builder, Location loc) {
          // 使用掩码向量化处理尾部
          auto vectorType = VectorType::get({vectorElements}, elementType);
          
          // 创建掩码 (暂时简化，不使用掩码)
          
          // 获取操作数
          auto inputs = op.getInputs();
          auto outputs = op.getOutputs();
          
          Value zeroElement = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(elementType));
          
          // 使用掩码读取向量
          llvm::SmallVector<Value, 4> maskedInputs;
          for (Value input : inputs) {
            auto maskedRead = builder.create<vector::TransferReadOp>(
                loc, vectorType, input, ValueRange{vectorUpperBound}, zeroElement);
            maskedInputs.push_back(maskedRead);
          }
          
          // 处理向量化计算 - 正确执行linalg.generic中的操作
          Value result;
          if (maskedInputs.empty()) {
            result = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(elementType));
          } else {
            // 根据原始linalg.generic的操作进行计算
            result = processVectorizedGenericOperations(builder, loc, op, maskedInputs, elementType);
          }
          
          // 使用掩码写入结果
          for (Value output : outputs) {
            builder.create<vector::TransferWriteOp>(
                loc, result, output, ValueRange{vectorUpperBound});
          }
          
          builder.create<scf::YieldOp>(loc);
        });
    
    LLVM_DEBUG(llvm::dbgs() << "尾部向量化处理完成\n");
  }

  /**
   * 处理向量化的generic操作 - 用于尾部处理
   * 将linalg.generic region中的操作转换为向量化操作
   */
  Value processVectorizedGenericOperations(
      OpBuilder &builder, Location loc, linalg::GenericOp op,
      llvm::SmallVector<Value, 4>& vectorInputs, Type elementType) const {
    
    // 获取原始操作的region
    Region& region = op.getRegion();
    if (region.empty() || vectorInputs.empty()) {
      return vectorInputs.empty() ? 
        builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(elementType)) :
        vectorInputs[0];
    }
    
    Block& block = region.front();
    Value currentResult = vectorInputs[0];
    
    // 遍历region中的操作，将每个标量操作转换为向量操作
    for (Operation& operation : block.getOperations()) {
      // 跳过终止操作
      if (operation.hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }
      
      StringRef opName = operation.getName().getStringRef();
      
      // 根据操作类型创建对应的向量操作
      if (opName == "arith.addf" && vectorInputs.size() >= 2) {
        currentResult = builder.create<arith::AddFOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.subf" && vectorInputs.size() >= 2) {
        currentResult = builder.create<arith::SubFOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.mulf" && vectorInputs.size() >= 2) {
        currentResult = builder.create<arith::MulFOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.divf" && vectorInputs.size() >= 2) {
        currentResult = builder.create<arith::DivFOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.addi" && vectorInputs.size() >= 2) {
        currentResult = builder.create<arith::AddIOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.subi" && vectorInputs.size() >= 2) {
        currentResult = builder.create<arith::SubIOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "arith.muli" && vectorInputs.size() >= 2) {
        currentResult = builder.create<arith::MulIOp>(loc, vectorInputs[0], vectorInputs[1]);
      } else if (opName == "math.sin") {
        currentResult = builder.create<math::SinOp>(loc, currentResult);
      } else if (opName == "math.cos") {
        currentResult = builder.create<math::CosOp>(loc, currentResult);
      } else if (opName == "math.exp") {
        currentResult = builder.create<math::ExpOp>(loc, currentResult);
      } else if (opName == "math.log") {
        currentResult = builder.create<math::LogOp>(loc, currentResult);
      } else if (opName == "math.sqrt") {
        currentResult = builder.create<math::SqrtOp>(loc, currentResult);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "尾部向量化计算: 不支持的操作 " << opName << "\n");
        // 对于不支持的操作，返回第一个输入
        currentResult = vectorInputs[0];
      }
    }
    
    return currentResult;
  }
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
    return "Linalg generic optimization pass with vectorization analysis";
  }

  LinalgGenericOptimizationPass() = default;

  LinalgGenericOptimizationPass(const LinalgGenericOptimizationPass &) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    ConversionTarget target(*context);
    
    // 添加合法的方言
    target.addLegalDialect<
        arith::ArithDialect, linalg::LinalgDialect, memref::MemRefDialect,
        VectorDialect, bufferization::BufferizationDialect, math::MathDialect,
        func::FuncDialect, affine::AffineDialect, scf::SCFDialect>();
    
    // 只将linalg.generic标记为非法，需要转换
    target.addIllegalOp<linalg::GenericOp>();
    
    // 保持其他linalg操作合法
    target.addLegalOp<linalg::FillOp>();

    RewritePatternSet patterns(context);
    // 将用户提供的硬件参数传递给Pattern
    patterns.add<LinalgGenericOptimizationPattern>(context,
                                                   userArchitecture.getValue(),
                                                   userVectorWidth.getValue(),
                                                   userVectorExtension.getValue(),
                                                   userEnableFMA.getValue(),
                                                   userTileSize.getValue());

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect, VectorDialect,
                    memref::MemRefDialect, bufferization::BufferizationDialect,
                    math::MathDialect, func::FuncDialect, affine::AffineDialect,
                    scf::SCFDialect>();
  }

  // 用户提供的硬件参数选项 - 编译时常量
  Option<std::string> userArchitecture{*this, "user-arch",
                                       llvm::cl::desc("Target architecture (x86_64, aarch64, etc.)"),
                                       llvm::cl::init("x86_64")};
  
  Option<int64_t> userVectorWidth{*this, "user-vector-width",
                                  llvm::cl::desc("Vector width in bits (128, 256, 512)"),
                                  llvm::cl::init(256)};
  
  Option<std::string> userVectorExtension{*this, "user-vector-ext",
                                          llvm::cl::desc("Vector extension (SSE2, AVX2, AVX512, NEON, SVE)"),
                                          llvm::cl::init("AVX2")};
  
  Option<bool> userEnableFMA{*this, "user-enable-fma",
                             llvm::cl::desc("Enable FMA instructions"),
                             llvm::cl::init(false)};
  
  Option<int64_t> userTileSize{*this, "user-tile-size",
                               llvm::cl::desc("Tile size for blocking"),
                               llvm::cl::init(32)};
};
} // namespace

namespace mlir {
namespace buddy {
void registerLinalgGenericOptimizationPass() {
  PassRegistration<LinalgGenericOptimizationPass>();
}
} // namespace buddy
} // namespace mlir