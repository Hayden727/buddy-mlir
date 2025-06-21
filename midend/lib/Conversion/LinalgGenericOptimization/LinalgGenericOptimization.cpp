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
#include <cassert>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <string>
#include <unordered_map>

using namespace mlir;
using namespace vector;
using namespace affine;
using namespace linalg;

//===----------------------------------------------------------------------===//
// LinalgGenericOptimizationPattern
//===----------------------------------------------------------------------===//

namespace {

class LinalgGenericOptimizationPattern : public OpConversionPattern<GenericOp> {
public:
  explicit LinalgGenericOptimizationPattern(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    return failure();
  }

private:
  std::unordered_map<Operation *, Operation *> isVectorizableOpsCache;
  // 标量arith和math操作到vector方言操作的映射表
  std::unordered_map<StringRef, StringRef> scalarToVectorOpMap = {
      // arith
      {"arith.addf", "vector.add"},
      {"arith.subf", "vector.sub"},
      {"arith.mulf", "vector.mul"},
      {"arith.divf", "vector.div"},
      {"arith.negf", "vector.neg"},
      {"arith.select", "vector.select"},
      {"arith.cmpf", "vector.cmp"},
      {"arith.constant", "vector.splat"},
      {"arith.index_cast", "vector.type_cast"},
      {"arith.trunci", "vector.trunc"},
      {"arith.uitofp", "vector.uitofp"},
      {"arith.sitofp", "vector.sitofp"},
      {"arith.cmpi", "vector.cmp"},
      {"arith.addi", "vector.add"},
      {"arith.subi", "vector.sub"},
      {"arith.muli", "vector.mul"},
      {"arith.divsi", "vector.div"},
      {"arith.divui", "vector.div"},
      {"arith.andi", "vector.and"},
      {"arith.ori", "vector.or"},
      {"arith.xori", "vector.xor"},
      // math
      {"math.cos", "math.cos"},
      {"math.sin", "math.sin"},
      {"math.rsqrt", "math.rsqrt"},
      {"math.fpowi", "math.fpowi"},
      {"math.exp", "math.exp"},
      {"math.log", "math.log"},
      {"math.tanh", "math.tanh"},
      {"math.absf", "math.absf"}
      // ... 可继续补充
  };

  // 迭代器类型枚举
  enum class IteratorType {
    Parallel,
    Reduction
  };

  // 提取迭代器类型
  std::unordered_map<int, IteratorType> extractIteratorTypes(GenericOp op) const {
    std::unordered_map<int, IteratorType> iteratorTypes;
    
    // 获取iterator_types属性
    ArrayAttr iteratorTypesAttr = op.getIteratorTypes();
    if (!iteratorTypesAttr) {
      return iteratorTypes; // 返回空映射
    }

    // 遍历iterator_types，提取每个迭代器的类型
    for (int i = 0; i < iteratorTypesAttr.size(); ++i) {
      Attribute attr = iteratorTypesAttr[i];
      StringRef iteratorType = cast<StringAttr>(attr).getValue();
      
      // 构建映射，记录迭代器索引及其类型
      if (iteratorType == "parallel") {
        iteratorTypes[i] = IteratorType::Parallel;
      } else if (iteratorType == "reduction") {
        iteratorTypes[i] = IteratorType::Reduction;
      }
      // 可以扩展支持其他迭代器类型
    }

    return iteratorTypes;
  }

  // 判断是否适合向量化的辅助方法
  bool isVectorizationSuitable(const std::unordered_map<int, IteratorType>& iteratorTypes) const {
    bool hasParallel = false;
    bool hasReduction = false;
    std::vector<int> parallelDims;
    std::vector<int> reductionDims;

    // 统计迭代器类型并记录维度
    for (const auto& pair : iteratorTypes) {
      if (pair.second == IteratorType::Parallel) {
        hasParallel = true;
        parallelDims.push_back(pair.first);
      } else if (pair.second == IteratorType::Reduction) {
        hasReduction = true;
        reductionDims.push_back(pair.first);
      }
    }

    // 决策逻辑：
    // 1. 如果存在至少一个"parallel"迭代器，则该维度可能适合向量化
    // 2. 如果存在"reduction"迭代器，则需要使用vector.reduction或vector.contract处理
    // 3. 如果所有迭代器都是"reduction"，向量化可能不适用
    if (!hasParallel) {
      return false; // 没有并行迭代器，不适合向量化
    }

    // 4. 优先对外层parallel维度进行分块向量化
    // 这样可以获得更好的数据局部性和缓存效率
    return true; // 有并行迭代器，适合向量化
  }

  // 获取建议的向量化策略
  struct VectorizationStrategy {
    std::vector<int> parallelDims;    // 并行维度
    std::vector<int> reductionDims;   // 归约维度
    bool shouldTile;                  // 是否应该分块
    int tileSize;                     // 建议的分块大小
  };

  VectorizationStrategy getVectorizationStrategy(const std::unordered_map<int, IteratorType>& iteratorTypes) const {
    VectorizationStrategy strategy;
    strategy.shouldTile = false;
    strategy.tileSize = 16; // 默认分块大小

    // 分离并行和归约维度
    for (const auto& pair : iteratorTypes) {
      if (pair.second == IteratorType::Parallel) {
        strategy.parallelDims.push_back(pair.first);
      } else if (pair.second == IteratorType::Reduction) {
        strategy.reductionDims.push_back(pair.first);
      }
    }

    // 如果有多个并行维度，建议分块
    if (strategy.parallelDims.size() >= 2) {
      strategy.shouldTile = true;
    }

    return strategy;
  }

  // 索引映射分析结果
  struct IndexMappingAnalysis {
    std::vector<AffineMap> inputMaps;     // 输入张量的索引映射
    std::vector<AffineMap> outputMaps;    // 输出张量的索引映射
    std::vector<bool> inputContinuity;    // 输入张量的连续性
    std::vector<bool> outputContinuity;   // 输出张量的连续性
    std::vector<int> bestVectorizationDims; // 建议的向量化维度
    bool hasTranspose;                    // 是否存在转置访问
    bool hasComplexMapping;               // 是否存在复杂映射
  };

  // 分析索引映射
  IndexMappingAnalysis analyzeIndexMappings(GenericOp op) const {
    IndexMappingAnalysis analysis;
    
    // 获取indexing_maps属性
    ArrayAttr indexingMapsAttr = op.getIndexingMaps();
    if (!indexingMapsAttr) {
      return analysis;
    }

    // 分离输入和输出映射
    size_t numInputs = op.getNumInputs();
    size_t numOutputs = op.getNumOutputs();
    
    // 提取输入映射
    for (size_t i = 0; i < numInputs; ++i) {
      AffineMap map = cast<AffineMapAttr>(indexingMapsAttr[i]).getValue();
      analysis.inputMaps.push_back(map);
      analysis.inputContinuity.push_back(isContinuousMapping(map));
    }
    
    // 提取输出映射
    for (size_t i = 0; i < numOutputs; ++i) {
      AffineMap map = cast<AffineMapAttr>(indexingMapsAttr[numInputs + i]).getValue();
      analysis.outputMaps.push_back(map);
      analysis.outputContinuity.push_back(isContinuousMapping(map));
    }

    // 分析映射特征
    analysis.hasTranspose = hasTransposeMapping(analysis.inputMaps, analysis.outputMaps);
    analysis.hasComplexMapping = hasComplexMapping(analysis.inputMaps, analysis.outputMaps);
    
    // 确定最佳向量化维度
    analysis.bestVectorizationDims = determineBestVectorizationDims(
        op, analysis.inputMaps, analysis.outputMaps, analysis.inputContinuity, analysis.outputContinuity);

    return analysis;
  }

  // 检查映射是否连续
  bool isContinuousMapping(AffineMap map) const {
    // 检查是否为恒等映射或简单的线性映射
    if (map.isIdentity()) {
      return true;
    }

    // 检查是否为简单的线性映射 (i, j) -> (i, j) 或 (i, j) -> (j, i)
    if (map.getNumDims() == 2 && map.getNumResults() == 2) {
      auto results = map.getResults();
      if (results.size() == 2) {
        // 检查是否为 (i, j) -> (i, j) 或 (i, j) -> (j, i)
        auto first = results[0].dyn_cast<AffineDimExpr>();
        auto second = results[1].dyn_cast<AffineDimExpr>();
        if (first && second) {
          return true; // 简单的维度重排
        }
      }
    }

    // 检查是否为简单的偏移映射 (i, j) -> (i + c, j + d)
    for (auto result : map.getResults()) {
      if (auto binOp = result.dyn_cast<AffineBinaryOpExpr>()) {
        if (binOp.getKind() == AffineExprKind::Add) {
          auto lhs = binOp.getLHS().dyn_cast<AffineDimExpr>();
          auto rhs = binOp.getRHS().dyn_cast<AffineConstantExpr>();
          if (lhs && rhs) {
            continue; // 简单的偏移，仍然连续
          }
        }
      }
      // 复杂的表达式，可能不连续
      if (!result.isa<AffineDimExpr>()) {
        return false;
      }
    }

    return true;
  }

  // 映射类型分类
  enum class MappingType {
    Identity,      // (i, j) -> (i, j) - 恒等映射
    Transpose,     // (i, j) -> (j, i) - 转置
    Broadcast,     // (i, j) -> (0, j) - 广播
    Projection,    // (i, j, k) -> (i, j) - 投影
    Permutation,   // (i, j, k) -> (i, k, j) - 维度重排
    Complex        // 复杂表达式
  };

  // 分析映射类型
  MappingType analyzeMappingType(AffineMap map) const {
    if (map.isIdentity()) {
      return MappingType::Identity;
    }

    auto results = map.getResults();
    int numDims = map.getNumDims();
    int numResults = map.getNumResults();

    // 检查广播映射 (如 #map = affine_map<(d0, d1) -> (0, d1)>)
    bool hasBroadcast = false;
    for (auto result : results) {
      if (auto constExpr = result.dyn_cast<AffineConstantExpr>()) {
        hasBroadcast = true;
      }
    }
    if (hasBroadcast) {
      return MappingType::Broadcast;
    }

    // 检查投影映射 (如 #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>)
    if (numResults < numDims) {
      return MappingType::Projection;
    }

    // 检查转置映射 (如 #map18 = affine_map<(d0, d1) -> (d1, d0)>)
    if (numDims == 2 && numResults == 2) {
      auto first = results[0].dyn_cast<AffineDimExpr>();
      auto second = results[1].dyn_cast<AffineDimExpr>();
      if (first && second && first.getPosition() == 1 && second.getPosition() == 0) {
        return MappingType::Transpose;
      }
    }

    // 检查维度重排 (如 #map10 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>)
    if (numDims == numResults && numDims > 2) {
      bool isPermutation = true;
      std::vector<int> usedDims(numDims, 0);
      for (auto result : results) {
        if (auto dimExpr = result.dyn_cast<AffineDimExpr>()) {
          usedDims[dimExpr.getPosition()]++;
        } else {
          isPermutation = false;
          break;
        }
      }
      // 检查每个维度是否只使用一次
      for (int count : usedDims) {
        if (count != 1) {
          isPermutation = false;
          break;
        }
      }
      if (isPermutation) {
        return MappingType::Permutation;
      }
    }

    // 检查复杂表达式
    for (auto result : results) {
      if (auto binOp = result.dyn_cast<AffineBinaryOpExpr>()) {
        if (binOp.getKind() == AffineExprKind::Mul) {
          return MappingType::Complex;
        }
        if (binOp.getKind() == AffineExprKind::Add) {
          auto lhs = binOp.getLHS().dyn_cast<AffineDimExpr>();
          auto rhs = binOp.getRHS().dyn_cast<AffineDimExpr>();
          if (lhs && rhs) {
            return MappingType::Complex; // 两个维度的加法
          }
        }
      }
    }

    return MappingType::Complex;
  }

  // 检查映射是否适合向量化
  bool isMappingVectorizable(AffineMap map) const {
    MappingType type = analyzeMappingType(map);
    
    switch (type) {
      case MappingType::Identity:
      case MappingType::Transpose:
      case MappingType::Broadcast:
      case MappingType::Projection:
      case MappingType::Permutation:
        return true; // 这些类型都适合向量化
      case MappingType::Complex:
        return false; // 复杂表达式可能不适合向量化
      default:
        return false;
    }
  }

  // 更新连续性检查方法
  bool isContinuousMapping(AffineMap map) const {
    MappingType type = analyzeMappingType(map);
    
    // 恒等映射和转置映射是连续的
    if (type == MappingType::Identity || type == MappingType::Transpose) {
      return true;
    }
    
    // 广播映射在某些情况下是连续的
    if (type == MappingType::Broadcast) {
      return isBroadcastContinuous(map);
    }
    
    // 投影和重排映射通常是连续的
    if (type == MappingType::Projection || type == MappingType::Permutation) {
      return true;
    }
    
    return false;
  }

  // 检查广播映射是否连续
  bool isBroadcastContinuous(AffineMap map) const {
    auto results = map.getResults();
    
    // 检查是否为简单的广播模式
    // 例如: (i, j) -> (0, j) 或 (i, j, k) -> (0, 0, k)
    for (size_t i = 0; i < results.size(); ++i) {
      auto result = results[i];
      if (auto constExpr = result.dyn_cast<AffineConstantExpr>()) {
        // 常量表达式（广播维度）
        continue;
      } else if (auto dimExpr = result.dyn_cast<AffineDimExpr>()) {
        // 维度表达式，检查是否在最后几个维度
        if (dimExpr.getPosition() != i) {
          return false; // 维度不匹配，可能不连续
        }
      } else {
        return false; // 复杂表达式
      }
    }
    
    return true;
  }

  // 检查是否存在转置映射
  bool hasTransposeMapping(const std::vector<AffineMap>& inputMaps, 
                          const std::vector<AffineMap>& outputMaps) const {
    for (const auto& map : inputMaps) {
      if (isTransposeMap(map)) {
        return true;
      }
    }
    for (const auto& map : outputMaps) {
      if (isTransposeMap(map)) {
        return true;
      }
    }
    return false;
  }

  // 检查是否为转置映射
  bool isTransposeMap(AffineMap map) const {
    if (map.getNumDims() == 2 && map.getNumResults() == 2) {
      auto results = map.getResults();
      if (results.size() == 2) {
        auto first = results[0].dyn_cast<AffineDimExpr>();
        auto second = results[1].dyn_cast<AffineDimExpr>();
        if (first && second && first.getPosition() == 1 && second.getPosition() == 0) {
          return true; // (i, j) -> (j, i)
        }
      }
    }
    return false;
  }

  // 检查是否存在复杂映射
  bool hasComplexMapping(const std::vector<AffineMap>& inputMaps, 
                        const std::vector<AffineMap>& outputMaps) const {
    for (const auto& map : inputMaps) {
      if (hasComplexExpressions(map)) {
        return true;
      }
    }
    for (const auto& map : outputMaps) {
      if (hasComplexExpressions(map)) {
        return true;
      }
    }
    return false;
  }

  // 检查映射是否包含复杂表达式
  bool hasComplexExpressions(AffineMap map) const {
    for (auto result : map.getResults()) {
      if (auto binOp = result.dyn_cast<AffineBinaryOpExpr>()) {
        if (binOp.getKind() == AffineExprKind::Mul) {
          return true; // 乘法表达式通常复杂
        }
        if (binOp.getKind() == AffineExprKind::Add) {
          auto lhs = binOp.getLHS().dyn_cast<AffineDimExpr>();
          auto rhs = binOp.getRHS().dyn_cast<AffineDimExpr>();
          if (lhs && rhs) {
            return true; // 两个维度的加法，可能复杂
          }
        }
      }
    }
    return false;
  }

  // 确定最佳向量化维度
  std::vector<int> determineBestVectorizationDims(GenericOp op,
                                                 const std::vector<AffineMap>& inputMaps,
                                                 const std::vector<AffineMap>& outputMaps,
                                                 const std::vector<bool>& inputContinuity,
                                                 const std::vector<bool>& outputContinuity) const {
    std::vector<int> bestDims;
    
    // 获取迭代器类型
    auto iteratorTypes = extractIteratorTypes(op);
    
    // 优先选择连续的并行维度进行向量化
    for (const auto& pair : iteratorTypes) {
      if (pair.second == IteratorType::Parallel) {
        int dim = pair.first;
        
        // 检查该维度在所有输入输出中是否连续
        bool isContinuousInAll = true;
        
        // 检查输入连续性
        for (size_t i = 0; i < inputMaps.size(); ++i) {
          if (!inputContinuity[i]) {
            // 检查这个特定输入在该维度是否连续
            if (!isDimensionContinuous(inputMaps[i], dim)) {
              isContinuousInAll = false;
              break;
            }
          }
        }
        
        // 检查输出连续性
        for (size_t i = 0; i < outputMaps.size(); ++i) {
          if (!outputContinuity[i]) {
            // 检查这个特定输出在该维度是否连续
            if (!isDimensionContinuous(outputMaps[i], dim)) {
              isContinuousInAll = false;
              break;
            }
          }
        }
        
        if (isContinuousInAll) {
          bestDims.push_back(dim);
        }
      }
    }
    
    return bestDims;
  }

  // 检查特定维度是否连续
  bool isDimensionContinuous(AffineMap map, int dim) const {
    if (map.isIdentity()) {
      return true;
    }
    
    // 检查该维度是否直接映射（没有复杂表达式）
    for (auto result : map.getResults()) {
      if (auto dimExpr = result.dyn_cast<AffineDimExpr>()) {
        if (dimExpr.getPosition() == dim) {
          return true; // 直接映射
        }
      }
    }
    
    return false;
  }

  StringRef getVectorOpName(Operation &op) const {
    StringRef opName = op.getName().getStringRef();
    auto it = scalarToVectorOpMap.find(opName);
    if (it != scalarToVectorOpMap.end()) {
      return it->second;
    }
    return StringRef(); // 返回空StringRef表示不可向量化
  }

  bool isVectorizableWithCache(Operation &op) const {
    auto it = isVectorizableOpsCache.find(&op);
    if (it != isVectorizableOpsCache.end()) {
        return it->second != nullptr; // 从缓存获取结果
    }
    
    // 执行实际的检查逻辑
    bool result = performVectorizationCheck(op);
    isVectorizableOpsCache[&op] = result ? &op : nullptr;
    return result;
  }

  bool performVectorizationCheck(Operation &op) const {
    // 检查操作是否在映射表中
    StringRef vectorOpName = getVectorOpName(op);
    if (vectorOpName.empty()) {
      return false; // 不在映射表中，不可向量化
    }

    // 检查操作的操作数类型是否支持向量化
    // 在linalg.generic的基本块中，操作的操作数通常是标量类型
    for (Value operand : op.getOperands()) {
      Type operandType = operand.getType();
      // 跳过已经是vector类型的操作数
      if (llvm::isa<VectorType>(operandType)) {
        continue;
      }
      // 检查标量类型是否支持向量化
      if (!operandType.isIntOrFloat() && !llvm::isa<IndexType>(operandType)) {
        return false; // 不支持的类型
      }
    }

    // 检查操作的结果类型是否支持向量化
    for (Type resultType : op.getResultTypes()) {
      // 跳过已经是vector类型的结果
      if (llvm::isa<VectorType>(resultType)) {
        continue;
      }
      // 检查标量类型是否支持向量化
      if (!resultType.isIntOrFloat() && !llvm::isa<IndexType>(resultType)) {
        return false; // 不支持的类型
      }
    }

    return true; // 通过所有检查，可以向量化
  }

  // 操作类型分类
  enum class OperationType {
    Scalar,     // 标量操作 (arith.addf, arith.mulf)
    Reduction,  // 归约操作 (累加、累乘等)
    Complex,    // 复杂操作 (分支、非仿射控制流)
    Terminator  // 终止操作 (linalg.yield)
  };

  // 区域内操作分析结果
  struct RegionAnalysis {
    std::vector<Operation*> scalarOps;      // 标量操作
    std::vector<Operation*> reductionOps;   // 归约操作
    std::vector<Operation*> complexOps;     // 复杂操作
    bool hasReduction;                      // 是否存在归约操作
    bool hasComplexFlow;                    // 是否存在复杂控制流
    bool allOpsVectorizable;                // 是否所有操作都可向量化
  };

  // 依赖类型
  enum class DependencyType {
    RAW,  // Read After Write (读后写)
    WAR,  // Write After Read (写后读)
    WAW   // Write After Write (写后写)
  };

  // 依赖关系
  struct Dependency {
    Operation* source;           // 源操作
    Operation* target;           // 目标操作
    DependencyType type;         // 依赖类型
    Value value;                 // 依赖的值
    bool isCrossIteration;       // 是否跨迭代依赖
    int iterationDim;            // 相关的迭代维度
  };

  // 依赖分析结果
  struct DependencyAnalysis {
    std::vector<Dependency> dependencies;   // 所有依赖关系
    bool hasCrossIterationDeps;             // 是否存在跨迭代依赖
    bool hasLoopCarriedDeps;                // 是否存在循环携带依赖
    bool isVectorizationSafe;               // 向量化是否安全
  };

  // 3. 分析区域内操作
  RegionAnalysis analyzeRegionOperations(GenericOp op) const {
    RegionAnalysis analysis;
    analysis.hasReduction = false;
    analysis.hasComplexFlow = false;
    analysis.allOpsVectorizable = true;

    // 获取region
    Region& region = op.getRegion();
    if (region.empty()) {
      return analysis;
    }

    Block& block = region.front();
    
    // 遍历region中的操作
    for (Operation& operation : block.getOperations()) {
      // 跳过终止操作
      if (operation.hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }

      // 分类操作
      OperationType opType = classifyOperation(operation);
      
      switch (opType) {
        case OperationType::Scalar:
          analysis.scalarOps.push_back(&operation);
          // 检查操作是否可向量化
          if (!isVectorizableWithCache(operation)) {
            analysis.allOpsVectorizable = false;
          }
          break;
          
        case OperationType::Reduction:
          analysis.reductionOps.push_back(&operation);
          analysis.hasReduction = true;
          // 归约操作需要特殊处理
          if (!isReductionVectorizable(operation, op)) {
            analysis.allOpsVectorizable = false;
          }
          break;
          
        case OperationType::Complex:
          analysis.complexOps.push_back(&operation);
          analysis.hasComplexFlow = true;
          analysis.allOpsVectorizable = false;
          break;
          
        case OperationType::Terminator:
          // 跳过终止操作
          break;
      }
    }

    return analysis;
  }

  // 分类操作类型
  OperationType classifyOperation(Operation& op) const {
    // 检查是否为终止操作
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      return OperationType::Terminator;
    }

    // 检查是否为标量操作
    StringRef opName = op.getName().getStringRef();
    if (opName.startswith("arith.") || opName.startswith("math.")) {
      // 检查是否为归约操作
      if (isReductionOperation(op)) {
        return OperationType::Reduction;
      }
      return OperationType::Scalar;
    }

    // 检查是否为复杂操作
    if (opName.startswith("cf.") || opName.startswith("scf.")) {
      return OperationType::Complex;
    }

    // 其他操作默认为复杂操作
    return OperationType::Complex;
  }

  // 检查是否为归约操作
  bool isReductionOperation(Operation& op) const {
    StringRef opName = op.getName().getStringRef();
    
    // 常见的归约操作
    if (opName == "arith.addf" || opName == "arith.addi" ||
        opName == "arith.mulf" || opName == "arith.muli" ||
        opName == "arith.maxf" || opName == "arith.maxi" ||
        opName == "arith.minf" || opName == "arith.mini") {
      return true;
    }
    
    return false;
  }

  // 检查归约操作是否可向量化
  bool isReductionVectorizable(Operation& op, GenericOp genericOp) const {
    // 获取迭代器类型
    auto iteratorTypes = extractIteratorTypes(genericOp);
    
    // 检查是否存在reduction迭代器
    bool hasReductionIterator = false;
    for (const auto& pair : iteratorTypes) {
      if (pair.second == IteratorType::Reduction) {
        hasReductionIterator = true;
        break;
      }
    }
    
    // 如果有reduction迭代器，归约操作可以向量化
    return hasReductionIterator;
  }

  // 4. 分析操作之间的数据依赖
  DependencyAnalysis analyzeDependencies(GenericOp op, const RegionAnalysis& regionAnalysis) const {
    DependencyAnalysis analysis;
    analysis.hasCrossIterationDeps = false;
    analysis.hasLoopCarriedDeps = false;
    analysis.isVectorizationSafe = true;

    // 获取迭代器类型
    auto iteratorTypes = extractIteratorTypes(op);
    
    // 构建依赖图
    std::vector<Operation*> allOps;
    allOps.insert(allOps.end(), regionAnalysis.scalarOps.begin(), regionAnalysis.scalarOps.end());
    allOps.insert(allOps.end(), regionAnalysis.reductionOps.begin(), regionAnalysis.reductionOps.end());
    
    // 分析操作间的依赖
    for (size_t i = 0; i < allOps.size(); ++i) {
      for (size_t j = i + 1; j < allOps.size(); ++j) {
        Operation* op1 = allOps[i];
        Operation* op2 = allOps[j];
        
        // 检查依赖关系
        auto deps = checkDependency(*op1, *op2, op);
        analysis.dependencies.insert(analysis.dependencies.end(), deps.begin(), deps.end());
      }
    }

    // 分析依赖特征
    for (const auto& dep : analysis.dependencies) {
      if (dep.isCrossIteration) {
        analysis.hasCrossIterationDeps = true;
        
        // 检查是否为循环携带依赖
        if (isLoopCarriedDependency(dep, iteratorTypes)) {
          analysis.hasLoopCarriedDeps = true;
          analysis.isVectorizationSafe = false;
        }
      }
    }

    return analysis;
  }

  // 检查两个操作之间的依赖关系
  std::vector<Dependency> checkDependency(Operation& op1, Operation& op2, GenericOp genericOp) const {
    std::vector<Dependency> deps;
    
    // 检查RAW依赖 (op2读op1写)
    for (Value result : op1.getResults()) {
      for (Value operand : op2.getOperands()) {
        if (result == operand) {
          Dependency dep;
          dep.source = &op1;
          dep.target = &op2;
          dep.type = DependencyType::RAW;
          dep.value = result;
          dep.isCrossIteration = false; // 默认值，后续分析
          dep.iterationDim = -1;
          deps.push_back(dep);
        }
      }
    }
    
    // 检查WAR依赖 (op1读op2写)
    for (Value result : op2.getResults()) {
      for (Value operand : op1.getOperands()) {
        if (result == operand) {
          Dependency dep;
          dep.source = &op2;
          dep.target = &op1;
          dep.type = DependencyType::WAR;
          dep.value = result;
          dep.isCrossIteration = false;
          dep.iterationDim = -1;
          deps.push_back(dep);
        }
      }
    }
    
    // 检查WAW依赖 (都写同一个值)
    for (Value result1 : op1.getResults()) {
      for (Value result2 : op2.getResults()) {
        if (result1 == result2) {
          Dependency dep;
          dep.source = &op1;
          dep.target = &op2;
          dep.type = DependencyType::WAW;
          dep.value = result1;
          dep.isCrossIteration = false;
          dep.iterationDim = -1;
          deps.push_back(dep);
        }
      }
    }
    
    return deps;
  }

  // 检查是否为循环携带依赖
  bool isLoopCarriedDependency(const Dependency& dep, 
                              const std::unordered_map<int, IteratorType>& iteratorTypes) const {
    // 这里需要更复杂的分析，检查依赖是否跨越迭代
    // 简化实现：检查是否涉及reduction迭代器
    for (const auto& pair : iteratorTypes) {
      if (pair.second == IteratorType::Reduction) {
        // 如果涉及reduction迭代器，可能是循环携带依赖
        return true;
      }
    }
    
    return false;
  }

  // 综合分析：结合映射关系、循环类型和操作依赖
  bool isComprehensiveVectorizable(GenericOp op) const {
    // 1. 提取迭代器类型
    auto iteratorTypes = extractIteratorTypes(op);
    if (!isVectorizationSuitable(iteratorTypes)) {
      return false;
    }

    // 2. 分析索引映射
    auto indexAnalysis = analyzeIndexMappings(op);
    if (indexAnalysis.bestVectorizationDims.empty()) {
      return false;
    }

    // 3. 分析区域内操作
    auto regionAnalysis = analyzeRegionOperations(op);
    if (!regionAnalysis.allOpsVectorizable) {
      return false;
    }

    // 4. 分析数据依赖
    auto depAnalysis = analyzeDependencies(op, regionAnalysis);
    if (!depAnalysis.isVectorizationSafe) {
      return false;
    }

    // 5. 检查连续性条件
    // 单操作向量化条件：操作可向量化 + 循环为parallel + 连续访存
    for (int dim : indexAnalysis.bestVectorizationDims) {
      auto it = iteratorTypes.find(dim);
      if (it == iteratorTypes.end() || it->second != IteratorType::Parallel) {
        continue; // 跳过非并行维度
      }
      
      // 检查该维度在所有输入输出中是否连续
      bool isContinuousInAll = true;
      for (const auto& map : indexAnalysis.inputMaps) {
        if (!isDimensionContinuous(map, dim)) {
          isContinuousInAll = false;
          break;
        }
      }
      for (const auto& map : indexAnalysis.outputMaps) {
        if (!isDimensionContinuous(map, dim)) {
          isContinuousInAll = false;
          break;
        }
      }
      
      if (isContinuousInAll) {
        return true; // 找到适合向量化的维度
      }
    }

    return false;
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
    return "Linalg generic optimization pass";
  }

  LinalgGenericOptimizationPass() = default;

  LinalgGenericOptimizationPass(const LinalgGenericOptimizationPass &) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    ConversionTarget target(*context);
    target.addLegalDialect<
        arith::ArithDialect, affine::AffineDialect, memref::MemRefDialect,
        VectorDialect, bufferization::BufferizationDialect, math::MathDialect,
        func::FuncDialect, linalg::LinalgDialect>();
    target
        .addIllegalOp<ModuleOp, func::FuncOp, func::ReturnOp, linalg::FillOp>();

    RewritePatternSet patterns(context);
    patterns.add<LinalgGenericOptimizationPattern>(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, affine::AffineDialect, VectorDialect,
                memref::MemRefDialect, bufferization::BufferizationDialect>();
  }

  // Option<int64_t> affineVectorSize{*this, "vector-size",
  //                                  llvm::cl::desc("Affine Vector size."),
  //                                  llvm::cl::init(16)};
};
} // namespace

namespace mlir {
namespace buddy {
void registerLinalgGenericOptimizationPass() {
  PassRegistration<LinalgGenericOptimizationPass>();
}
} // namespace buddy
} // namespace mlir