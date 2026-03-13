#!/bin/bash
# download-deps.sh
# 在在线环境中运行此脚本下载所有系统依赖包

set -e

echo "=== 下载Zephyr系统依赖包 ==="

DEPS_DIR="../deps"
mkdir -p $DEPS_DIR

echo "1. 更新包列表..."
sudo apt-get update

echo "2. 下载依赖包..."
sudo apt-get install -y --download-only \
    qemu-system-x86 \
    qemu-system-arm \
    qemu-system-misc \
    qemu-utils \
    gdb-multiarch \
    build-essential \
    git \
    cmake \
    ninja-build \
    python3-pip \
    python3-venv \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libpython3-dev \
    device-tree-compiler \
    libglib2.0-dev \
    libpixman-1-dev \
    libfdt-dev \
    libaio-dev \
    libusb-1.0-0-dev \
    libcapstone-dev

echo "3. 复制deb包到deps目录..."
sudo cp /var/cache/apt/archives/*.deb $DEPS_DIR/

echo "4. 生成包清单..."
ls -la $DEPS_DIR/*.deb > $DEPS_DIR/package-list.txt
echo "包数量: $(ls $DEPS_DIR/*.deb | wc -l)" >> $DEPS_DIR/package-list.txt

echo "5. 清理缓存..."
sudo apt-get clean

echo "=== 依赖包下载完成 ==="
echo "目录: $DEPS_DIR"
echo "总大小: $(du -sh $DEPS_DIR | cut -f1)"