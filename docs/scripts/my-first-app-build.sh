#!/bin/bash
# My First Zephyr App 构建脚本

set -e

echo "=== My First Zephyr App 构建脚本 ==="

# 默认板型
BOARD=${1:-qemu_x86}

# 检查是否在Zephyr环境中
if [ -z "$ZEPHYR_BASE" ]; then
    echo "错误: Zephyr环境未设置"
    echo "请先运行: source ~/.zephyr-env.sh"
    exit 1
fi

echo "板型: $BOARD"
echo "Zephyr目录: $ZEPHYR_BASE"
echo "当前目录: $(pwd)"

# 清理之前的构建
if [ -d "build" ]; then
    echo "清理之前的构建..."
    rm -rf build
fi

# 构建应用
echo "构建应用..."
west build -p auto -b $BOARD .

if [ $? -eq 0 ]; then
    echo "✅ 构建成功"
    
    # 显示构建信息
    echo ""
    echo "=== 构建信息 ==="
    echo "输出文件: build/zephyr/zephyr.elf"
    echo "输出文件: build/zephyr/zephyr.bin"
    echo "输出文件: build/zephyr/zephyr.hex"
    
    # 显示内存使用情况
    if [ -f "build/zephyr/zephyr.map" ]; then
        echo ""
        echo "=== 内存使用 ==="
        grep -A5 "Memory Configuration" build/zephyr/zephyr.map || true
    fi
    
    # 显示符号表大小
    if command -v size >/dev/null 2>&1; then
        echo ""
        echo "=== 文件大小 ==="
        size build/zephyr/zephyr.elf 2>/dev/null || true
    fi
else
    echo "❌ 构建失败"
    exit 1
fi

echo ""
echo "=== 运行选项 ==="
echo "1. 在QEMU中运行: west build -t run"
echo "2. 调试模式: west build -t debugserver"
echo "3. 清理构建: west build -t pristine"
echo ""
echo "构建完成!"