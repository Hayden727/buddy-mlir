#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'
BLUE='\033[0;34m'

# 检查命令行参数
if [ $# -ne 1 ]; then
    echo -e "${YELLOW}Usage: $0 <operator_name>${NC}"
    echo "Example: $0 'gemm' or $0 'conv2d' or $0 'axpy'"
    echo "Available operators: gemm, conv2d, axpy, dot-product, batch-mat-vec, batch-gemm, outer-product, softmax"
    exit 1
fi

OPERATOR="$1"
RUNS=100

# 构建4个命令名称
CMD_GPP="test-generic-${OPERATOR}-g++-run"
CMD_CLANGPP="test-generic-${OPERATOR}-clang++-run"
CMD_RUN="test-generic-${OPERATOR}-run"
CMD_MANUAL="test-generic-${OPERATOR}-manual-run"

# 创建临时文件存储输出
OUTPUT_GPP=$(mktemp)
OUTPUT_CLANGPP=$(mktemp)
OUTPUT_RUN=$(mktemp)
OUTPUT_MANUAL=$(mktemp)
PROCESSED_GPP=$(mktemp)
PROCESSED_CLANGPP=$(mktemp)
PROCESSED_RUN=$(mktemp)
PROCESSED_MANUAL=$(mktemp)
SPEEDUPS_GPP_VS_RUN=$(mktemp)
SPEEDUPS_CLANGPP_VS_RUN=$(mktemp)
SPEEDUPS_MANUAL_VS_RUN=$(mktemp)

# 提取时间数据（取最后一个匹配的数值，通常是执行时间）
extract_time() {
    local file="$1"
    grep -o '[0-9]\+\.[0-9]\+e[-+]\?[0-9]\+\|[0-9]\+\.[0-9]\+' "$file" | tail -1
}

# 转换时间为秒
convert_to_seconds() {
    local time_val="$1"
    if [[ $time_val =~ e ]]; then
        # 使用awk处理科学计数法，bc不能很好处理
        echo "$time_val" | awk '{printf "%.9f", $1}'
    else
        printf "%.9f" $time_val
    fi
}

# 计算平均值
calculate_mean() {
    local file="$1"
    local sum=0
    local count=0
    while read -r line; do
        sum=$(echo "$sum + $line" | bc -l)
        count=$((count + 1))
    done < "$file"
    if [ $count -gt 0 ]; then
        echo "scale=9; $sum / $count" | bc -l
    else
        echo "0"
    fi
}

echo -e "${BLUE}Running ${OPERATOR} operator with 4 implementations $RUNS times each...${NC}"

# 运行四个命令并计算每次的加速比
for ((i=1; i<=$RUNS; i++)); do
    echo -ne "\rRun $i/$RUNS"
    
    # 运行 g++ 版本
    TEMP_OUT_GPP=$(mktemp)
    make $CMD_GPP > "$TEMP_OUT_GPP" 2>/dev/null
    TIME_GPP=$(extract_time "$TEMP_OUT_GPP")
    if [ -n "$TIME_GPP" ]; then
        TIME_GPP=$(convert_to_seconds "$TIME_GPP")
    fi
    
    # 运行 clang++ 版本
    TEMP_OUT_CLANGPP=$(mktemp)
    make $CMD_CLANGPP > "$TEMP_OUT_CLANGPP" 2>/dev/null
    TIME_CLANGPP=$(extract_time "$TEMP_OUT_CLANGPP")
    if [ -n "$TIME_CLANGPP" ]; then
        TIME_CLANGPP=$(convert_to_seconds "$TIME_CLANGPP")
    fi
    
    # 运行普通 MLIR 版本
    TEMP_OUT_RUN=$(mktemp)
    make $CMD_RUN > "$TEMP_OUT_RUN" 2>/dev/null
    TIME_RUN=$(extract_time "$TEMP_OUT_RUN")
    if [ -n "$TIME_RUN" ]; then
        TIME_RUN=$(convert_to_seconds "$TIME_RUN")
    fi
    
    # 运行手动优化 MLIR 版本
    TEMP_OUT_MANUAL=$(mktemp)
    make $CMD_MANUAL > "$TEMP_OUT_MANUAL" 2>/dev/null
    TIME_MANUAL=$(extract_time "$TEMP_OUT_MANUAL")
    if [ -n "$TIME_MANUAL" ]; then
        TIME_MANUAL=$(convert_to_seconds "$TIME_MANUAL")
    fi
    
    # 首次运行时显示提取的时间用于调试
    if [ $i -eq 1 ]; then
        echo -e "\n${YELLOW}Debug: First run extracted times:${NC}"
        echo "G++ ($CMD_GPP): $TIME_GPP seconds"
        echo "Clang++ ($CMD_CLANGPP): $TIME_CLANGPP seconds"
        echo "MLIR Run ($CMD_RUN): $TIME_RUN seconds"
        echo "MLIR Manual ($CMD_MANUAL): $TIME_MANUAL seconds"
    fi
    
    # 保存第一次运行的输出用于比较
    if [ $i -eq 1 ]; then
        # 只提取 data = 后面的实际数据内容，排除最后一行的时间数值
        grep "data =" "$TEMP_OUT_GPP" | sed 's/.*data = //' > "$PROCESSED_GPP"
        grep "data =" "$TEMP_OUT_CLANGPP" | sed 's/.*data = //' > "$PROCESSED_CLANGPP"
        grep "data =" "$TEMP_OUT_RUN" | sed 's/.*data = //' > "$PROCESSED_RUN"
        grep "data =" "$TEMP_OUT_MANUAL" | sed 's/.*data = //' > "$PROCESSED_MANUAL"
        # 移除最后一行（时间信息）
        sed -i '$d' "$PROCESSED_GPP"
        sed -i '$d' "$PROCESSED_CLANGPP"
        sed -i '$d' "$PROCESSED_RUN"
        sed -i '$d' "$PROCESSED_MANUAL"
    fi
    
    # 计算这次运行的加速比（以MLIR Run为基准）
    if [ -n "$TIME_GPP" ] && [ -n "$TIME_RUN" ] && [ "$TIME_GPP" != "0" ] && [ "$TIME_RUN" != "0" ]; then
        echo "scale=9; $TIME_RUN/$TIME_GPP" | bc -l >> "$SPEEDUPS_GPP_VS_RUN"
    fi
    if [ -n "$TIME_CLANGPP" ] && [ -n "$TIME_RUN" ] && [ "$TIME_CLANGPP" != "0" ] && [ "$TIME_RUN" != "0" ]; then
        echo "scale=9; $TIME_RUN/$TIME_CLANGPP" | bc -l >> "$SPEEDUPS_CLANGPP_VS_RUN"
    fi
    if [ -n "$TIME_MANUAL" ] && [ -n "$TIME_RUN" ] && [ "$TIME_MANUAL" != "0" ] && [ "$TIME_RUN" != "0" ]; then
        echo "scale=9; $TIME_RUN/$TIME_MANUAL" | bc -l >> "$SPEEDUPS_MANUAL_VS_RUN"
    fi
    
    rm "$TEMP_OUT_GPP" "$TEMP_OUT_CLANGPP" "$TEMP_OUT_RUN" "$TEMP_OUT_MANUAL"
done
echo

# 比较数据输出
echo -e "\n${BLUE}Comparing output data across all implementations:${NC}"

# 比较所有实现的结果是否一致（以Baseline为基准）
ALL_MATCH=true
if ! diff "$PROCESSED_GPP" "$PROCESSED_RUN" > /dev/null; then
    echo -e "${RED}✗ G++ and Baseline outputs differ!${NC}"
    ALL_MATCH=false
fi
if ! diff "$PROCESSED_CLANGPP" "$PROCESSED_RUN" > /dev/null; then
    echo -e "${RED}✗ Clang++ and Baseline outputs differ!${NC}"
    ALL_MATCH=false
fi
if ! diff "$PROCESSED_MANUAL" "$PROCESSED_RUN" > /dev/null; then
    echo -e "${RED}✗ Optimization and Baseline outputs differ!${NC}"
    ALL_MATCH=false
fi

if [ "$ALL_MATCH" = true ]; then
    echo -e "${GREEN}✓ All implementations produce the same results as Baseline!${NC}"
else
    echo -e "${RED}✗ Found differences between implementations!${NC}"
fi

# 计算加速比的均值（以Baseline为基准）
echo -e "\n${BLUE}Performance Comparison (vs Baseline):${NC}"

# G++ vs Baseline
SPEEDUP_GPP_VS_RUN_MEAN=$(calculate_mean "$SPEEDUPS_GPP_VS_RUN")
if [ -n "$SPEEDUP_GPP_VS_RUN_MEAN" ] && [ "$SPEEDUP_GPP_VS_RUN_MEAN" != "0" ]; then
    if [ $(echo "$SPEEDUP_GPP_VS_RUN_MEAN > 1" | bc -l) -eq 1 ]; then
        printf "${GREEN}G++ is %.2fx faster than Baseline${NC}\n" "$SPEEDUP_GPP_VS_RUN_MEAN"
    else
        printf "${RED}G++ is %.2f of Baseline${NC}\n" "$SPEEDUP_GPP_VS_RUN_MEAN"
    fi
fi

# Clang++ vs Baseline
SPEEDUP_CLANGPP_VS_RUN_MEAN=$(calculate_mean "$SPEEDUPS_CLANGPP_VS_RUN")
if [ -n "$SPEEDUP_CLANGPP_VS_RUN_MEAN" ] && [ "$SPEEDUP_CLANGPP_VS_RUN_MEAN" != "0" ]; then
    if [ $(echo "$SPEEDUP_CLANGPP_VS_RUN_MEAN > 1" | bc -l) -eq 1 ]; then
        printf "${GREEN}Clang++ is %.2fx faster than Baseline${NC}\n" "$SPEEDUP_CLANGPP_VS_RUN_MEAN"
    else
        printf "${RED}Clang++ is %.2f of Baseline${NC}\n" "$SPEEDUP_CLANGPP_VS_RUN_MEAN"
    fi
fi

# Optimization vs Baseline
SPEEDUP_MANUAL_VS_RUN_MEAN=$(calculate_mean "$SPEEDUPS_MANUAL_VS_RUN")
if [ -n "$SPEEDUP_MANUAL_VS_RUN_MEAN" ] && [ "$SPEEDUP_MANUAL_VS_RUN_MEAN" != "0" ]; then
    if [ $(echo "$SPEEDUP_MANUAL_VS_RUN_MEAN > 1" | bc -l) -eq 1 ]; then
        printf "${GREEN}Optimization is %.2fx faster than Baseline${NC}\n" "$SPEEDUP_MANUAL_VS_RUN_MEAN"
    else
        printf "${RED}Optimization is %.2f of Baseline${NC}\n" "$SPEEDUP_MANUAL_VS_RUN_MEAN"
    fi
fi

# 清理临时文件
rm "$OUTPUT_GPP" "$OUTPUT_CLANGPP" "$OUTPUT_RUN" "$OUTPUT_MANUAL" \
   "$PROCESSED_GPP" "$PROCESSED_CLANGPP" "$PROCESSED_RUN" "$PROCESSED_MANUAL" \
   "$SPEEDUPS_GPP_VS_RUN" "$SPEEDUPS_CLANGPP_VS_RUN" "$SPEEDUPS_MANUAL_VS_RUN"