# juzidaxia 博客 🍊

这是我的个人技术博客，使用 MkDocs + Material 主题构建，部署在 GitHub Pages。

## 🚀 快速开始

### 环境要求
- Python 3.8+
- pip

### 安装依赖
```bash
pip install mkdocs
pip install mkdocs-material
pip install mkdocs-git-revision-date-plugin
pip install mkdocs-git-committers
pip install mkdocs-minify-plugin
```

### 本地运行
```bash
# 启动本地开发服务器
python -m mkdocs serve

# 构建静态网站
python -m mkdocs build
```

访问 http://localhost:8000 查看效果。

## 📁 项目结构
```
juzidaxia/
├── docs/                    # 文档目录
│   ├── index.md            # 首页
│   ├── about.md            # 关于页面
│   ├── tech/               # 技术文章
│   ├── others/             # 其他文章
│   ├── _assets/            # 静态资源
│   ├── stylesheets/        # 自定义样式
│   └── javascripts/        # 自定义脚本
├── mkdocs.yml              # 配置文件
├── .github/workflows/      # GitHub Actions
└── README.MD               # 项目说明
```

## 🛠️ 开发指南

### 添加新文章
1. 在 `docs/` 目录下创建 Markdown 文件
2. 在 `mkdocs.yml` 的 `nav` 部分添加导航
3. 提交并推送更改

### 自定义样式
- 修改 `docs/stylesheets/extra.css`
- 修改 `docs/javascripts/extra.js`
- 主题配置在 `mkdocs.yml` 的 `theme` 部分

### 部署
博客通过 GitHub Actions 自动部署到 GitHub Pages。每次推送到 `main` 分支都会触发自动构建和部署。

## 🔧 功能特性

### 已实现功能
- ✅ 响应式设计，支持移动端
- ✅ 亮色/暗色主题切换
- ✅ 全文搜索
- ✅ 代码高亮和复制
- ✅ 文章阅读时间估算
- ✅ 图片灯箱效果
- ✅ 平滑滚动
- ✅ 自动部署

### 计划功能
- 🔄 评论系统 (Giscus)
- 🔄 访问统计 (Google Analytics/Umami)
- 🔄 SEO优化
- 🔄 文章分类标签

## 📄 许可证
本博客内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可协议。

## 🤝 贡献
欢迎提交 Issue 和 Pull Request！

## 📞 联系
- GitHub: [EWYan](https://github.com/EWYan)
- 邮箱: juzivis@qq.com
- 博客: https://ewyan.github.io/juzidaxia/

---

> **莫问前程，但行好事。保持好奇，持续学习。**