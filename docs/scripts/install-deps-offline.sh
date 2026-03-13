#!/bin/bash
# install-deps-offline.sh
# 在离线环境中安装系统依赖包

set -e

echo "=== 离线安装系统依赖包 ==="

DEPS_DIR="../deps"

if [ ! -d "$DEPS_DIR" ]; then
    echo "错误: deps目录不存在"
    exit 1
fi

echo "1. 检查deb包数量..."
DEB_COUNT=$(ls $DEPS_DIR/*.deb 2>/dev/null | wc -l)
if [ $DEB_COUNT -eq 0 ]; then
    echo "错误: 未找到deb包"
    exit 1
fi
echo "找到 $DEB_COUNT 个deb包"

echo "2. 安装所有deb包..."
for deb in $DEPS_DIR/*.deb; do
    echo "安装: $(basename $deb)"
    sudo dpkg -i "$deb" 2>/dev/null || true
done

echo "3. 修复依赖关系..."
sudo apt-get install -f -y

echo "4. 验证安装..."
echo "QEMU版本: $(qemu-system-x86_64 --version | head -1)"
echo "GDB版本: $(gdb-multiarch --version | head -1)"
echo "CMake版本: $(cmake --version | head -1)"
echo "Ninja版本: $(ninja --version)"

echo "=== 依赖包安装完成 ==="