# Buddy MLIR Linalg Generic Optimization Tests

本目录包含了用于测试 `linalg.generic` 操作向量化优化 Pass 的测试用例。

## 文件说明

### 测试文件
- `linalg-generic-optimization-test.mlir` - 基于 tensor 的测试用例
- `linalg-generic-memref-test.mlir` - 基于 memref 的测试用例
- `makefile` - 构建和运行测试的脚本

### 测试用例类型

#### 1. 基于 Tensor 的测试用例
包含以下7个测试场景：
- **简单元素级加法** - 测试基本的并行维度识别
- **二维矩阵元素级操作** - 测试多维并行
- **向量内积** - 测试 reduction 操作
- **矩阵行求和** - 测试混合迭代器类型
- **复杂的广播操作** - 测试内存访问模式
- **三维张量操作** - 测试高维度分析
- **多输入复杂计算** - 测试操作复杂度分析

#### 2. 基于 MemRef 的测试用例
包含以下7个测试场景：
- **简单元素级加法 (memref)** - 测试基本的并行维度识别
- **二维矩阵元素级操作 (memref)** - 测试多维并行
- **向量内积 (memref)** - 测试 reduction 操作
- **矩阵行求和 (memref)** - 测试混合迭代器类型
- **复杂的广播操作 (memref)** - 测试内存访问模式
- **转置操作 (memref)** - 测试非对角内存访问
- **动态维度测试 (memref)** - 测试未知大小处理

## 使用方法

### 前置条件
确保已经构建了 Buddy MLIR 项目：
```bash
# 在项目根目录
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir
ninja
```

### 运行测试

#### 1. 查看可用的测试选项
```bash
cd examples/BuddyGeneric
make help
```

#### 2. 运行完整的测试（包含执行）
```bash
# 运行基于 tensor 的测试
make generic-optimization-tensor-test

# 运行基于 memref 的测试  
make generic-optimization-memref-test

# 运行所有测试
make all-tests
```

#### 3. 只运行分析（不执行代码）
```bash
# 只运行 tensor 版本的分析
make generic-optimization-analysis-only

# 只运行 memref 版本的分析
make generic-optimization-memref-analysis-only

# 运行所有分析
make all-analysis
```

#### 4. 清理生成的文件
```bash
make clean
```

## 测试验证

### 分析阶段验证
运行 `*-analysis-only` 目标时，你应该看到：
- **硬件能力分析**：检测到的向量宽度（如 AVX2: 256 bits, AVX512: 512 bits）
- **操作模式分析**：迭代器类型分类（parallel, reduction）
- **内存访问分析**：访问模式识别（连续、步长、广播等）
- **向量化可行性判断**：最终的向量化决策

### 执行阶段验证
运行完整测试时，你应该看到：
- 正确的计算结果输出
- 执行时间统计
- FileCheck 验证通过

## 预期输出示例

### 分析输出示例
```
=== Linalg Generic Optimization Analysis ===
硬件能力分析: 检测到 AVX2 支持，向量宽度 256 bits
操作模式分析: 发现 1 个并行维度，0 个 reduction 维度
内存访问分析: 检测到连续内存访问模式
向量化决策: 建议进行向量化优化
```

### 执行结果示例
```
Unranked Memref base@ = 0x... rank = 1 offset = 0 sizes = [1024] strides = [1] data =
[5, 5, 5, 5, ...]
执行时间: 0.001234 秒
```

## 测试重点

### 1. 硬件能力分析
- 验证不同 SIMD 指令集的检测（SSE, AVX, AVX2, AVX512）
- 验证向量宽度计算的正确性

### 2. 操作模式分析  
- 验证并行维度的正确识别
- 验证 reduction 维度的正确识别
- 验证混合迭代器类型的处理

### 3. 内存访问分析
- 验证连续访问模式的识别
- 验证步长访问模式的识别
- 验证广播模式的识别
- 验证转置等复杂访问模式的识别

### 4. 向量化决策
- 验证向量化可行性判断的准确性
- 验证不同场景下的决策逻辑

## 故障排除

### 常见问题
1. **编译错误**：确保 LLVM/MLIR 已正确构建
2. **Pass 未找到**：确保 `linalg-generic-optimization` Pass 已正确注册
3. **执行失败**：检查共享库路径是否正确

### 调试技巧
1. 使用 `-mlir-print-ir-after-all` 查看 Pass 执行后的 IR
2. 使用 `-mlir-print-debuginfo` 获取更详细的调试信息
3. 逐步运行 Pass pipeline 定位问题

## 扩展测试

### 添加新的测试用例
1. 在 `.mlir` 文件中添加新的测试函数
2. 在 `main()` 函数中调用新的测试
3. 添加相应的 FileCheck 验证
4. 更新 makefile（如需要）

### 自定义测试参数
可以通过修改测试用例中的常量来测试不同的数据大小和计算模式：
```mlir
%c_size = arith.constant 2048 : index  // 修改数据大小
%c_value = arith.constant 3.14 : f32   // 修改测试数据
``` 