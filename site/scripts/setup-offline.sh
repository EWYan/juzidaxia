#!/bin/bash
# setup-offline.sh
# 离线环境完整安装脚本

set -e

echo "=========================================="
echo "    Zephyr RTOS离线环境安装脚本"
echo "=========================================="

WORK_DIR=$(pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$WORK_DIR/install-log-$TIMESTAMP.txt"

exec 2>&1 | tee "$LOG_FILE"

echo "安装开始时间: $(date)"
echo "工作目录: $WORK_DIR"
echo "日志文件: $LOG_FILE"

# 检查目录结构
echo ""
echo "=== 检查目录结构 ==="

REQUIRED_DIRS="deps sdk zephyr-src python-deps"
for dir in $REQUIRED_DIRS; do
    if [ -d "$dir" ]; then
        echo "✅ $dir 目录存在"
    else
        echo "❌ $dir 目录不存在"
        exit 1
    fi
done

# 检查必要文件
echo ""
echo "=== 检查必要文件 ==="

REQUIRED_FILES="
deps/*.deb
sdk/zephyr-sdk-0.17.1_linux-x86_64.tar.xz
zephyr-src-complete.tar.gz
python-deps/*.whl
"

for pattern in $REQUIRED_FILES; do
    if ls $pattern >/dev/null 2>&1; then
        echo "✅ $pattern 文件存在"
    else
        echo "❌ $pattern 文件不存在"
        exit 1
    fi
done

# 安装系统依赖
echo ""
echo "=== 安装系统依赖 ==="

echo "1. 安装deb包..."
for deb in deps/*.deb; do
    echo "  安装: $(basename $deb)"
    sudo dpkg -i "$deb" 2>/dev/null || true
done

echo "2. 修复依赖关系..."
sudo apt-get install -f -y

echo "3. 验证安装..."
echo "  QEMU: $(qemu-system-x86_64 --version | head -1)"
echo "  GDB: $(gdb-multiarch --version | head -1)"
echo "  CMake: $(cmake --version | head -1)"

# 安装Zephyr SDK
echo ""
echo "=== 安装Zephyr SDK ==="

echo "1. 解压SDK..."
sudo tar xf sdk/zephyr-sdk-0.17.1_linux-x86_64.tar.xz -C /opt/

ZEPHYR_SDK_INSTALL_DIR="/opt/zephyr-sdk-0.17.1"
if [ ! -d "$ZEPHYR_SDK_INSTALL_DIR" ]; then
    echo "❌ SDK解压失败"
    exit 1
fi

echo "2. 运行SDK安装脚本..."
cd "$ZEPHYR_SDK_INSTALL_DIR"
sudo ./setup.sh -t all
cd "$WORK_DIR"

echo "3. 验证SDK..."
if [ -f "$ZEPHYR_SDK_INSTALL_DIR/sdk_version" ]; then
    echo "✅ SDK安装成功: $(cat $ZEPHYR_SDK_INSTALL_DIR/sdk_version)"
else
    echo "❌ SDK安装失败"
    exit 1
fi

# 解压Zephyr源码
echo ""
echo "=== 解压Zephyr源码 ==="

echo "1. 解压源码..."
tar -xzf zephyr-src-complete.tar.gz

echo "2. 进入Zephyr目录..."
cd zephyr-src/zephyr
ZEPHYR_BASE=$(pwd)

# 设置Python环境
echo ""
echo "=== 设置Python环境 ==="

echo "1. 创建虚拟环境..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "2. 离线安装Python依赖..."
echo "  安装requirements.txt..."
pip install --no-index --find-links=../../python-deps/ -r scripts/requirements.txt

echo "  安装其他requirements..."
for req in base run-test extras compliance; do
    if [ -f "scripts/requirements-$req.txt" ]; then
        echo "  安装requirements-$req.txt..."
        pip install --no-index --find-links=../../python-deps/ -r "scripts/requirements-$req.txt"
    fi
done

echo "  安装west工具..."
pip install --no-index --find-links=../../python-deps/ west

echo "3. 验证Python环境..."
echo "  Python: $(python --version)"
echo "  pip: $(pip --version | head -1)"
echo "  west: $(west --version)"

# 设置环境变量
echo ""
echo "=== 设置环境变量 ==="

ENV_FILE="$HOME/.zephyr-env.sh"
cat > "$ENV_FILE" << EOF
#!/bin/bash
# Zephyr环境变量设置

export ZEPHYR_SDK_INSTALL_DIR="$ZEPHYR_SDK_INSTALL_DIR"
export ZEPHYR_BASE="$ZEPHYR_BASE"
export PATH="\$ZEPHYR_SDK_INSTALL_DIR/bin:\$PATH"

# 激活虚拟环境
if [ -f "\$ZEPHYR_BASE/.venv/bin/activate" ]; then
    source "\$ZEPHYR_BASE/.venv/bin/activate"
fi

# 加载Zephyr环境
if [ -f "\$ZEPHYR_BASE/zephyr-env.sh" ]; then
    source "\$ZEPHYR_BASE/zephyr-env.sh"
fi
EOF

chmod +x "$ENV_FILE"

# 添加到bashrc
if ! grep -q "zephyr-env.sh" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Zephyr环境" >> ~/.bashrc
    echo "source $ENV_FILE" >> ~/.bashrc
fi

echo "环境变量文件: $ENV_FILE"

# 测试构建
echo ""
echo "=== 测试构建 ==="

source "$ENV_FILE"

echo "1. 构建hello_world示例..."
west build -p auto -b qemu_x86 samples/hello_world

if [ $? -eq 0 ]; then
    echo "✅ 构建成功"
else
    echo "❌ 构建失败"
    exit 1
fi

echo "2. 测试QEMU运行..."
timeout 10 west build -t run > /tmp/qemu-test.log 2>&1 &
QEMU_PID=$!
sleep 3

if ps -p $QEMU_PID > /dev/null; then
    echo "✅ QEMU启动成功"
    kill $QEMU_PID 2>/dev/null
else
    echo "❌ QEMU启动失败"
    cat /tmp/qemu-test.log
    exit 1
fi

# 创建快捷脚本
echo ""
echo "=== 创建快捷脚本 ==="

cat > "$WORK_DIR/zephyr-quick-start.sh" << 'EOF'
#!/bin/bash
# Zephyr快速启动脚本

source ~/.zephyr-env.sh

echo "Zephyr环境已激活"
echo "ZEPHYR_BASE: $ZEPHYR_BASE"
echo "ZEPHYR_SDK_INSTALL_DIR: $ZEPHYR_SDK_INSTALL_DIR"
echo "Python: $(python --version)"
echo "west: $(west --version)"

cd "$ZEPHYR_BASE"
exec bash
EOF

chmod +x "$WORK_DIR/zephyr-quick-start.sh"

echo "快速启动脚本: $WORK_DIR/zephyr-quick-start.sh"

# 完成
echo ""
echo "=========================================="
echo "    Zephyr离线环境安装完成！"
echo "=========================================="
echo ""
echo "✅ 安装完成时间: $(date)"
echo ""
echo "📋 安装摘要:"
echo "  - 系统依赖: 已安装"
echo "  - Zephyr SDK: 已安装 ($ZEPHYR_SDK_INSTALL_DIR)"
echo "  - Zephyr源码: 已解压 ($ZEPHYR_BASE)"
echo "  - Python环境: 已配置"
echo "  - 环境变量: 已设置"
echo "  - 构建测试: 通过"
echo ""
echo "🚀 使用方法:"
echo "  1. 重新打开终端或运行: source ~/.bashrc"
echo "  2. 或使用快捷脚本: ./zephyr-quick-start.sh"
echo ""
echo "🔧 测试命令:"
echo "  west build -p auto -b qemu_x86 samples/hello_world"
echo "  west build -t run"
echo "  west build -t debugserver"
echo ""
echo "📝 安装日志: $LOG_FILE"
echo "=========================================="