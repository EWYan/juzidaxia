# Zephyr调试环境脚本集

本文档包含Zephyr RTOS调试环境搭建所需的所有脚本文件。这些脚本与[Zephyr RTOS调试环境搭建完全指南](../tech/zephyr_debug_env.md)文章配套使用。

## 📋 脚本列表

### 1. 系统依赖管理
- [`download-deps.sh`](download-deps.sh) - 下载系统依赖包
- [`install-deps-offline.sh`](install-deps-offline.sh) - 离线安装系统依赖

### 2. Zephyr SDK管理
- [`download-sdk.sh`](download-sdk.sh) - 下载Zephyr SDK
- [`install-sdk.sh`](install-sdk.sh) - 安装Zephyr SDK

### 3. 源码和依赖管理
- [`download-zephyr-src.sh`](download-zephyr-src.sh) - 下载Zephyr源码
- [`download-python-deps.sh`](download-python-deps.sh) - 下载Python依赖包

### 4. 完整安装脚本
- [`setup-offline.sh`](setup-offline.sh) - 一键离线安装脚本

### 5. 示例应用
- [`my-first-app-build.sh`](my-first-app-build.sh) - 示例应用构建脚本

## 🚀 使用流程

### 在线环境准备
1. 下载所有脚本到 `scripts/` 目录
2. 运行 `download-*.sh` 脚本下载所需文件
3. 在有网络的环境中准备好所有材料

### 离线环境部署
1. 将整个 `scripts/` 目录和下载的文件复制到目标机器
2. 运行 `setup-offline.sh` 进行一键安装
3. 验证安装：`west build -p auto -b qemu_x86 samples/hello_world`

## 📝 脚本说明

### 设计原则
1. **原子性**：每个脚本完成一个明确的任务
2. **可重入**：脚本可以安全地多次运行
3. **错误处理**：提供清晰的错误信息和恢复建议
4. **日志记录**：详细的执行日志便于调试
5. **进度显示**：实时显示安装进度

### 兼容性
- 操作系统：Ubuntu 20.04/22.04 LTS 或兼容系统
- 架构：x86_64
- 依赖：Python 3.8+, Git, curl/wget

## 🔧 自定义修改

你可以根据实际需求修改这些脚本：

1. **调整版本**：修改脚本中的版本号变量
2. **更改下载源**：替换镜像源提高下载速度
3. **添加新功能**：根据项目需求扩展脚本功能
4. **优化错误处理**：根据实际环境调整错误处理逻辑

## 📄 许可证

这些脚本采用 MIT 许可证，你可以自由使用、修改和分发。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这些脚本：

1. **报告问题**：描述遇到的问题和环境信息
2. **提交修复**：通过 Pull Request 提交代码改进
3. **添加功能**：扩展脚本功能或添加新的脚本
4. **改进文档**：完善使用说明和注意事项

## 🔗 相关资源

- [Zephyr官方文档](https://docs.zephyrproject.org/)
- [Zephyr GitHub仓库](https://github.com/zephyrproject-rtos/zephyr)
- [完整教程文章](../tech/zephyr_debug_env.md)

---

> **让嵌入式开发更简单，让环境搭建不再困难。** 🚀