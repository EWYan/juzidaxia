#!/bin/bash
# install-sdk.sh
# 安装Zephyr SDK

set -e

echo "=== 安装Zephyr SDK ==="

SDK_DIR="../sdk"
SDK_VERSION="0.17.1"
SDK_FILE="zephyr-sdk-${SDK_VERSION}_linux-x86_64.tar.xz"
ZEPHYR_SDK_INSTALL_DIR="/opt/zephyr-sdk-${SDK_VERSION}"

if [ ! -f "$SDK_DIR/$SDK_FILE" ]; then
    echo "错误: SDK文件不存在: $SDK_DIR/$SDK_FILE"
    exit 1
fi

echo "1. 验证SDK文件..."
if ! tar -tf "$SDK_DIR/$SDK_FILE" > /dev/null 2>&1; then
    echo "❌ SDK文件损坏"
    exit 1
fi

echo "2. 解压SDK到 /opt..."
sudo tar xf "$SDK_DIR/$SDK_FILE" -C /opt/

if [ ! -d "$ZEPHYR_SDK_INSTALL_DIR" ]; then
    echo "错误: SDK解压失败"
    exit 1
fi

echo "3. 运行SDK安装脚本..."
cd "$ZEPHYR_SDK_INSTALL_DIR"
sudo ./setup.sh -t all

echo "4. 设置环境变量..."
cat >> ~/.bashrc << EOF

# Zephyr SDK环境变量
export ZEPHYR_SDK_INSTALL_DIR="$ZEPHYR_SDK_INSTALL_DIR"
EOF

echo "5. 验证安装..."
if [ -f "$ZEPHYR_SDK_INSTALL_DIR/sdk_version" ]; then
    echo "✅ SDK安装成功"
    echo "SDK版本: $(cat $ZEPHYR_SDK_INSTALL_DIR/sdk_version)"
else
    echo "❌ SDK安装失败"
    exit 1
fi

echo "6. 测试工具链..."
TOOLS_TO_TEST="arm-zephyr-eabi-gcc riscv64-zephyr-elf-gcc x86_64-zephyr-elf-gcc"
for tool in $TOOLS_TO_TEST; do
    if command -v $tool > /dev/null 2>&1; then
        echo "✅ $tool 可用"
    else
        # 检查SDK目录中的工具
        if [ -f "$ZEPHYR_SDK_INSTALL_DIR/$tool" ] || [ -f "$ZEPHYR_SDK_INSTALL_DIR/bin/$tool" ]; then
            echo "✅ $tool 在SDK中"
        else
            echo "⚠️  $tool 未找到"
        fi
    fi
done

echo "=== SDK安装完成 ==="
echo "安装目录: $ZEPHYR_SDK_INSTALL_DIR"
echo "请运行: source ~/.bashrc 加载环境变量"