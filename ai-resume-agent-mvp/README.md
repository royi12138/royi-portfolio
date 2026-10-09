# AI 岗位匹配简历助手 MVP 0.2

本版本在原有文件解析基础上，加入阿里云百炼（千问）的简历分析接口。它仍是开发中的 MVP，不代表完整上线产品。

## 当前功能
1. PDF / DOCX / JPG / PNG 简历上传，限制 10MB、3 页
2. PDF / DOCX 文本解析；图片 OCR 需要系统安装 Tesseract 和中文语言包
3. Resume JSON 结构化展示
4. 原始临时文件在解析完成后清理
5. 用户勾选确认后，才将简历文本发送给模型分析
6. 调用模型前尝试隐藏手机号、邮箱和身份证号
7. AI 分析结果包含优点、原文引用、问题原因、修改方向和补充真实信息的问题
8. 后端只从环境变量读取 API Key，不将 Key 写入前端或源码

## 本地运行
建议使用 Python 3.11+。

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python run.py

在运行程序前，先在当前终端设置环境变量：

    export DASHSCOPE_API_KEY="在本地终端粘贴你自己的新Key"
    export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    export DASHSCOPE_MODEL="qwen-plus"
    python run.py

浏览器访问 http://127.0.0.1:8000 。

不要把真实 API Key 提交到 GitHub，也不要把密钥发到聊天里。`.env` 已列在 `.gitignore` 中；`.env.example` 只包含变量名和示例地址，不包含真实密钥。当前代码不会自动读取 `.env` 文件。

## 百炼配置说明
默认使用华北2（北京）的兼容接口地址：

    https://dashscope.aliyuncs.com/compatible-mode/v1

API Key 的地域/计费方案必须与 Base URL 匹配。如果 Key 是在其他地域或专属业务空间创建的，请以百炼控制台及官方 Base URL 文档为准，修改 `DASHSCOPE_BASE_URL`。不要混用不同地域的 Key 和 URL。

环境变量：
- `DASHSCOPE_API_KEY`：必填，真实 API Key
- `DASHSCOPE_BASE_URL`：可选，默认使用上方北京地址
- `DASHSCOPE_MODEL`：可选，默认 `qwen-plus`

官方文档：
- API Key 与 Base URL：https://help.aliyun.com/zh/model-studio/get-api-key
- Base URL 总览：https://help.aliyun.com/zh/model-studio/base-url
- 千问 OpenAI 兼容 Chat 接口：https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions

## 隐私与事实保护说明
- 简历解析上传的原始文件会在解析完成后从临时目录清理。
- 只有用户勾选并点击“开始 AI 分析”时，浏览器才会将解析得到的文本发送给后端；后端会先尝试隐藏手机号、邮箱和身份证号，再向模型服务发送文字。
- 这是简单的正则脱敏，不能保证识别所有个人信息。姓名、公司名称、地址或其他细节仍可能留在文本里；请勿上传你不愿发送给模型服务的内容。
- 本版本的事实保护依靠系统提示词、原文引用校验和用户核对，不是完美保证。模型输出应视为建议，不可未经核实直接用于求职。
- 当前没有用户账号、持久化简历存储、微信登录/支付、岗位 JD 分析、逐条接受/拒绝、Word 导出或生产部署配置。

## API
- `GET /api/health`：健康检查
- `POST /api/resume/parse`：上传简历文件并解析
- `POST /api/resume/analyze`：接收 JSON `{ "resume_text": "..." }` 并返回 AI 分析

## 目录
    ai-resume-agent-mvp/
      app/
        ai_analysis.py
        main.py
        __init__.py
        static/index.html
      demo/index.html
      requirements.txt
      run.py
      .env.example
      README.md
