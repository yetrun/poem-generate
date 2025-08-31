---
title: LSTM Poetry (Gradio)
emoji: 📜
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# LSTM Poetry (Gradio)

五言绝句小应用（Gradio）。本仓库可直接用于 **Hugging Face Spaces** 自动部署。

## 一键部署（推荐）

**方式 A：在网页端创建 Space 并上传代码**
1. 登录 Hugging Face，创建 Space：SDK 选 **Gradio**，硬件选 **CPU Basic (FREE)**，可选公开（Public）。
2. 在 Space 页面 → *Files* → *Upload files*，上传：
   - `app.py`
   - `requirements.txt`
   - `runtime.txt`
   - `.gitattributes`
   - `poetry_vocabulary.txt`
   - `lstm_poetry_model.keras`（>100MB 请用 LFS）
3. 提交后自动构建，完成即上线。

**方式 B：连接 GitHub 仓库**
1. 在你的 GitHub 仓库放入本模板文件。
2. 仓库启用 Git LFS：
   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes lstm_poetry_model.keras
   git commit -m "Add model via LFS"
   git push
   ```