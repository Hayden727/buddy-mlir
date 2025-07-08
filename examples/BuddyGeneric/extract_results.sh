#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'
BLUE='\033[0;34m'

# 检查result.log文件是否存在
if [ ! -f "result.log" ]; then
    echo -e "${RED}Error: result.log file not found!${NC}"
    echo "Please run bench.sh first to generate the results."
    exit 1
fi

echo -e "${BLUE}Extracting benchmark results from result.log...${NC}"
echo

# 创建临时文件存储提取的结果
TEMP_RESULTS=$(mktemp)

# 提取算子名称和性能对比结果
current_operator=""
while IFS= read -r line; do
    # 去除ANSI颜色代码
    clean_line=$(echo "$line" | sed 's/\x1b\[[0-9;]*m//g')
    
    # 提取算子名称
    if [[ $clean_line =~ Running[[:space:]]([a-zA-Z-]+)[[:space:]]operator ]]; then
        current_operator="${BASH_REMATCH[1]}"
        echo "=== ${current_operator^^} ===" >> "$TEMP_RESULTS"
    fi
    
    # 提取性能对比结果中的加速比
    if [[ $clean_line =~ ^G\+\+ ]]; then
        if [[ $clean_line =~ ([0-9]+\.[0-9]+)x ]]; then
            speedup="${BASH_REMATCH[1]}"
            echo "G++: ${speedup}x" >> "$TEMP_RESULTS"
        elif [[ $clean_line =~ ([0-9]+\.[0-9]+)[[:space:]]of ]]; then
            speedup="${BASH_REMATCH[1]}"
            echo "G++: ${speedup} of Baseline" >> "$TEMP_RESULTS"
        fi
    elif [[ $clean_line =~ ^Clang\+\+ ]]; then
        if [[ $clean_line =~ ([0-9]+\.[0-9]+)x ]]; then
            speedup="${BASH_REMATCH[1]}"
            echo "Clang++: ${speedup}x" >> "$TEMP_RESULTS"
        elif [[ $clean_line =~ ([0-9]+\.[0-9]+)[[:space:]]of ]]; then
            speedup="${BASH_REMATCH[1]}"
            echo "Clang++: ${speedup} of Baseline" >> "$TEMP_RESULTS"
        fi
    elif [[ $clean_line =~ ^Optimization ]]; then
        if [[ $clean_line =~ ([0-9]+\.[0-9]+)x ]]; then
            speedup="${BASH_REMATCH[1]}"
            echo "Optimization: ${speedup}x" >> "$TEMP_RESULTS"
        elif [[ $clean_line =~ ([0-9]+\.[0-9]+)[[:space:]]of ]]; then
            speedup="${BASH_REMATCH[1]}"
            echo "Optimization: ${speedup} of Baseline" >> "$TEMP_RESULTS"
        fi
    fi
    
    # 如果遇到空行且当前有算子，说明一个算子的结果结束
    if [[ -z "$clean_line" && -n "$current_operator" ]]; then
        echo "" >> "$TEMP_RESULTS"
        current_operator=""
    fi
done < result.log

# 显示提取的结果
cat "$TEMP_RESULTS"

# 将结果追加到result.log
echo "" >> result.log
echo "=== EXTRACTED SPEEDUP RESULTS ===" >> result.log
cat "$TEMP_RESULTS" >> result.log

# 清理临时文件
rm "$TEMP_RESULTS"

echo -e "${BLUE}Results extraction completed and saved to result.log!${NC}" 