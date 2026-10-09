# AI 岗位匹配简历助手 MVP 0.1

这是第一阶段可运行版本。

当前重点：
1. PDF / DOCX / JPG / PNG 简历上传
2. 文件类型、大小、页数校验
3. PDF / DOCX 文本解析
4. 图片 OCR
5. Resume JSON 结构化
6. 原始临时文件解析后自动清理
7. 浏览器展示解析结果
8. 独立的静态交互 Demo

本地运行建议使用 Python 3.11+：

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py

浏览器打开：
http://127.0.0.1:8000

OCR 使用 pytesseract。Mac 可以安装 tesseract 和中文语言包。

当前不包含：
- 真实大模型调用
- 简历缺点分析 Agent
- JD 图片 OCR/分析 Agent
- 简历-JD 匹配
- Fact Guard
- 微信登录
- 微信支付
- Word 生成
- PostgreSQL
- 生产环境部署

目录：
ai-resume-agent-mvp/
  app/
    main.py
    __init__.py
    static/index.html
  demo/index.html
  requirements.txt
  run.py
  README.md

demo/index.html 是交互原型；app/static/index.html 是连接真实上传解析 API 的 MVP 页面。
