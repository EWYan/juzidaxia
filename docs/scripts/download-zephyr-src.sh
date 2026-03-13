#!/bin/bash
# download-zephyr-src.sh
# 下载Zephyr源码

set -e

echo "=== 下载Zephyr源码 ==="

ZEPHYR_SRC_DIR="../zephyr-src"
mkdir -p $ZEPHYR_SRC_DIR
cd $ZEPHYR_SRC_DIR

echo "1. 克隆Zephyr主仓库..."
if [ ! -d "zephyr" ]; then
    git clone --depth=1 https://github.com/zephyrproject-rtos/zephyr.git
else
    echo "Zephyr目录已存在，跳过克隆"
fi

cd zephyr

echo "2. 创建Python虚拟环境..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "3. 安装west工具..."
pip install west

echo "4. 初始化west工作区..."
west init -l .

echo "5. 更新所有模块..."
west update

echo "6. 记录版本信息..."
git log --oneline -5 > version-info.txt
echo "Zephyr版本: $(git describe --always)" >> version-info.txt
echo "下载时间: $(date)" >> version-info.txt

echo "7. 生成源码清单..."
find . -type f -name "*.c" -o -name "*.h" -o -name "*.py" -o -name "*.cmake" -o -name "CMakeLists.txt" -o -name "*.txt" | \
    sort > file-list.txt
echo "文件总数: $(wc -l < file-list.txt)" >> version-info.txt

echo "8. 打包源码..."
cd ../..
echo "正在打包源码，这可能需要几分钟..."
tar -czf zephyr-src-complete.tar.gz zephyr-src/

echo "9. 生成校验文件..."
md5sum zephyr-src-complete.tar.gz > zephyr-src-complete.tar.gz.md5
sha256sum zephyr-src-complete.tar.gz > zephyr-src-complete.tar.gz.sha256

echo "=== Zephyr源码下载完成 ==="
echo "源码目录: $ZEPHYR_SRC_DIR"
echo "打包文件: zephyr-src-complete.tar.gz"
echo "文件大小: $(du -h zephyr-src-complete.tar.gz | cut -f1)"
echo "MD5: $(cat zephyr-src-complete.tar.gz.md5)"
echo "SHA256: $(cat zephyr-src-complete.tar.gz.sha256)"