#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'
BLUE='\033[0;34m'

# 检查命令行参数
if [ $# -ne 2 ]; then
    echo -e "${YELLOW}Usage: $0 <command1> <command2>${NC}"
    echo "Example: $0 'next-reduce-sum-run' 'next-reduce-sum-vec-manual-run'"
    exit 1
fi

CMD1="$1"
CMD2="$2"
RUNS=10

# 创建临时文件存储输出
OUTPUT1=$(mktemp)
OUTPUT2=$(mktemp)
PROCESSED1=$(mktemp)
SPEEDUPS=$(mktemp)

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

echo -e "${BLUE}Running each version $RUNS times...${NC}"

# 运行两个命令并计算每次的加速比
for ((i=1; i<=$RUNS; i++)); do
    echo -ne "\rRun $i/$RUNS"
    
    # 运行第一个命令
    TEMP_OUT1=$(mktemp)
    make $CMD1 > "$TEMP_OUT1" 2>/dev/null
    TIME1=$(extract_time "$TEMP_OUT1")
    if [ -n "$TIME1" ]; then
        TIME1=$(convert_to_seconds "$TIME1")
    fi
    
    # 运行第二个命令
    TEMP_OUT2=$(mktemp)
    make $CMD2 > "$TEMP_OUT2" 2>/dev/null
    TIME2=$(extract_time "$TEMP_OUT2")
    if [ -n "$TIME2" ]; then
        TIME2=$(convert_to_seconds "$TIME2")
    fi
    
    # 首次运行时显示提取的时间用于调试
    if [ $i -eq 1 ]; then
        echo -e "\n${YELLOW}Debug: First run extracted times:${NC}"
        echo "CMD1 ($CMD1): $TIME1 seconds"
        echo "CMD2 ($CMD2): $TIME2 seconds"
    fi
    
    # 保存第一次运行的输出用于比较
    if [ $i -eq 1 ]; then
        # 只提取 data = 后面的实际数据内容，排除最后一行的时间数值
        grep "data =" "$TEMP_OUT1" | sed 's/.*data = //' > "$PROCESSED1"
        grep "data =" "$TEMP_OUT2" | sed 's/.*data = //' > "$OUTPUT2"
        # 移除最后一行（时间信息）
        sed -i '$d' "$PROCESSED1"
        sed -i '$d' "$OUTPUT2"
    fi
    
    # 计算这次运行的加速比
    if [ -n "$TIME1" ] && [ -n "$TIME2" ] && [ "$TIME1" != "0" ] && [ "$TIME2" != "0" ]; then
        echo "scale=9; $TIME1/$TIME2" | bc -l >> "$SPEEDUPS"
    fi
    
    rm "$TEMP_OUT1" "$TEMP_OUT2"
done
echo

# 比较数据输出
echo -e "\n${BLUE}Comparing output data:${NC}"
if diff "$PROCESSED1" "$OUTPUT2" > /dev/null; then
    echo -e "${GREEN}✓ Outputs match! Both versions produce the same results.${NC}"
else
    echo -e "${RED}✗ Outputs differ! Found differences:${NC}"
    echo "----------------------------------------"
    diff "$PROCESSED1" "$OUTPUT2"
    echo "----------------------------------------"
fi

# 计算加速比的均值
echo -e "\n${BLUE}Performance Comparison:${NC}"
SPEEDUP_MEAN=$(calculate_mean "$SPEEDUPS")

if [ -n "$SPEEDUP_MEAN" ] && [ "$SPEEDUP_MEAN" != "0" ]; then
    if [ $(echo "$SPEEDUP_MEAN > 1" | bc -l) -eq 1 ]; then
        printf "${GREEN}Second version is %.2fx faster${NC}\n" "$SPEEDUP_MEAN"
    else
        SLOWDOWN=$(echo "scale=2; 1/$SPEEDUP_MEAN" | bc -l)
        printf "${RED}Second version is %.2fx slower${NC}\n" "$SLOWDOWN"
    fi
fi

# 清理临时文件
rm "$OUTPUT1" "$OUTPUT2" "$PROCESSED1" "$SPEEDUPS"