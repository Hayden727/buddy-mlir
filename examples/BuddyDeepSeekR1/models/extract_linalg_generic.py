#!/usr/bin/env python3
"""
提取MLIR文件中linalg.generic操作的脚本
"""

import sys
from pathlib import Path

def extract_linalg_generic_ops(input_file, output_file):
    """
    从MLIR文件中提取linalg.generic操作
    
    Args:
        input_file: 输入的MLIR文件路径
        output_file: 输出的文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    linalg_ops = []
    i = 0
    while i < len(lines):
        # 移除前导空格来检查 `linalg.generic`
        if lines[i].lstrip().startswith('linalg.generic'):
            op_lines = []
            brace_level = 0
            
            # 找到了 linalg.generic，开始收集
            for j in range(i, len(lines)):
                line = lines[j]
                op_lines.append(line.rstrip())
                
                brace_level += line.count('{')
                brace_level -= line.count('}')
                
                # `linalg.generic` 自身可能没有花括号，但我们已经开始收集了
                # 当花括号平衡且我们已经看到至少一个开花括号时，操作结束
                if brace_level == 0 and any('{' in l for l in op_lines):
                    linalg_ops.append('\n'.join(op_lines))
                    i = j + 1
                    break
            else: # for 循环未 break
                # 如果文件结束但花括号不平衡，这也是一种结束
                i = j + 1
        else:
            i += 1

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"// 从 {input_file} 中提取的 linalg.generic 操作\n")
        f.write(f"// 总共找到 {len(linalg_ops)} 个操作\n\n")
        
        for idx, op in enumerate(linalg_ops, 1):
            f.write(f"// 操作 #{idx}:\n")
            f.write(op)
            f.write("\n\n")
    
    print(f"成功提取了 {len(linalg_ops)} 个 linalg.generic 操作到 {output_file}")
    return linalg_ops

def main():
    if len(sys.argv) != 3:
        print("用法: python extract_linalg_generic.py <input_file> <output_file>")
        print("示例: python extract_linalg_generic.py subgraph0-lower.mlir linalg_ops.mlir")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"错误: 输入文件 {input_file} 不存在")
        sys.exit(1)
    
    try:
        # 使用提取函数
        ops = extract_linalg_generic_ops(input_file, output_file)
        
        if ops:
            print(f"提取完成！共找到 {len(ops)} 个 linalg.generic 操作")
            print(f"结果已保存到: {output_file}")
        else:
            print("未找到任何 linalg.generic 操作")
            
    except Exception as e:
        print(f"提取过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 