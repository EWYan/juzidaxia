#!/bin/bash
# download-python-deps.sh
# 下载Python依赖包

set -e

echo "=== 下载Python依赖包 ==="

PYTHON_DEPS_DIR="../python-deps"
mkdir -p $PYTHON_DEPS_DIR

ZEPHYR_SRC_DIR="../zephyr-src/zephyr"
if [ ! -d "$ZEPHYR_SRC_DIR" ]; then
    echo "错误: Zephyr源码目录不存在"
    exit 1
fi

cd $ZEPHYR_SRC_DIR

echo "1. 激活虚拟环境..."
if [ ! -d ".venv" ]; then
    echo "错误: 虚拟环境不存在"
    exit 1
fi
source .venv/bin/activate

echo "2. 下载requirements.txt依赖..."
REQUIREMENT_FILES="
scripts/requirements.txt
scripts/requirements-base.txt
scripts/requirements-run-test.txt
scripts/requirements-extras.txt
scripts/requirements-compliance.txt
"

for req_file in $REQUIREMENT_FILES; do
    if [ -f "$req_file" ]; then
        echo "下载: $req_file"
        pip download -r "$req_file" -d "../../$PYTHON_DEPS_DIR" --no-deps
    else
        echo "警告: $req_file 不存在"
    fi
done

echo "3. 下载west工具..."
pip download west -d "../../$PYTHON_DEPS_DIR" --no-deps

echo "4. 下载常用工具..."
TOOLS="
pyocd
spsdk
pytest
pylint
mypy
coverage
gcovr
"

for tool in $TOOLS; do
    echo "下载: $tool"
    pip download "$tool" -d "../../$PYTHON_DEPS_DIR" --no-deps 2>/dev/null || echo "警告: $tool 下载失败"
done

echo "5. 生成依赖清单..."
cd "../../$PYTHON_DEPS_DIR"
ls -la *.whl *.tar.gz *.zip 2>/dev/null | sort > package-list.txt

echo "6. 统计信息..."
TOTAL_FILES=$(ls *.whl *.tar.gz *.zip 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh . | cut -f1)

cat > deps-info.txt << EOF
Python依赖包信息
================
下载时间: $(date)
文件数量: $TOTAL_FILES
总大小: $TOTAL_SIZE
包含的requirements文件:
$(for req_file in $REQUIREMENT_FILES; do echo "  - $req_file"; done)
EOF

echo "=== Python依赖包下载完成 ==="
echo "目录: $PYTHON_DEPS_DIR"
echo "文件数量: $TOTAL_FILES"
echo "总大小: $TOTAL_SIZE"
echo "清单文件: package-list.txt"