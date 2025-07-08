#!/bin/bash

# 清空或创建result.log文件
> result.log

# 运行所有算子测试并保存结果
# echo "Running all operator benchmarks..." | tee result.log

# 使用tee命令同时输出到终端和文件
./compare_performence.sh gemm 2>&1 | tee -a result.log
./compare_performence.sh conv2d 2>&1 | tee -a result.log
./compare_performence.sh axpy 2>&1 | tee -a result.log
./compare_performence.sh dot-product 2>&1 | tee -a result.log
./compare_performence.sh batch-mat-vec 2>&1 | tee -a result.log
./compare_performence.sh batch-gemm 2>&1 | tee -a result.log
./compare_performence.sh outer-product 2>&1 | tee -a result.log
./compare_performence.sh softmax 2>&1 | tee -a result.log

echo "Benchmark completed. Results saved to result.log"
echo "Run './extract_results.sh' to view formatted results"