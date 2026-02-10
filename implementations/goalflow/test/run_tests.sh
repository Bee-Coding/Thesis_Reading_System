#!/bin/bash

# GoalFlowMatcher 测试运行脚本

echo "========================================"
echo "GoalFlowMatcher 测试套件"
echo "========================================"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到 Python"
    exit 1
fi

echo "Python 版本:"
python --version
echo ""

# 检查 PyTorch
echo "检查 PyTorch..."
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 错误: 未安装 PyTorch"
    exit 1
fi
echo ""

# 运行测试
echo "开始运行测试..."
echo "========================================"
python test_goal_flow_matcher.py

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 所有测试通过！"
    echo "========================================"
    exit 0
else
    echo ""
    echo "========================================"
    echo "❌ 测试失败"
    echo "========================================"
    exit 1
fi
