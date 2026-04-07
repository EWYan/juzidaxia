#!/bin/bash
# download-sdk.sh
# 下载Zephyr SDK

set -e

echo "=== 下载Zephyr SDK ==="

SDK_DIR="../sdk"
mkdir -p $SDK_DIR
cd $SDK_DIR

SDK_VERSION="0.17.1"
SDK_FILE="zephyr-sdk-${SDK_VERSION}_linux-x86_64.tar.xz"
SDK_URL="https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v${SDK_VERSION}/${SDK_FILE}"

echo "1. 下载SDK文件..."
wget $SDK_URL
wget ${SDK_URL}.sha256

echo "2. 验证文件完整性..."
if sha256sum -c ${SDK_FILE}.sha256; then
    echo "✅ SDK文件完整性验证通过"
else
    echo "❌ SDK文件完整性验证失败"
    exit 1
fi

echo "3. 解压测试..."
if tar -tf $SDK_FILE > /dev/null 2>&1; then
    echo "✅ SDK文件格式正确"
else
    echo "❌ SDK文件损坏"
    exit 1
fi

echo "4. 计算文件信息..."
echo "SDK文件: $SDK_FILE"
echo "文件大小: $(du -h $SDK_FILE | cut -f1)"
echo "MD5: $(md5sum $SDK_FILE | cut -d' ' -f1)"
echo "SHA256: $(sha256sum $SDK_FILE | cut -d' ' -f1)"

cat > sdk-info.txt << EOF
Zephyr SDK信息
===============
版本: ${SDK_VERSION}
文件: ${SDK_FILE}
下载时间: $(date)
下载URL: ${SDK_URL}
文件大小: $(du -h $SDK_FILE | cut -f1)
MD5: $(md5sum $SDK_FILE | cut -d' ' -f1)
SHA256: $(sha256sum $SDK_FILE | cut -d' ' -f1)
EOF

echo "=== SDK下载完成 ==="
echo "文件保存在: $SDK_DIR/$SDK_FILE"
echo "详细信息: $SDK_DIR/sdk-info.txt"