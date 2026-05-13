# ai-rag-demo

一个基于 FastAPI、LangChain、Chroma 和 DeepSeek/Qwen Embedding 的简历 RAG 问答示例项目。项目支持上传 PDF、TXT、DOCX、CSV 文档，将文档切分并写入本地向量库，然后按用户维度进行简历问答、流式聊天、来源追溯和文档删除。

## 功能概览

- 文档上传：支持 `.pdf`、`.txt`、`.docx`、`.csv`
- 文档去重：基于 `file_hash + user_id` 避免同一用户重复上传
- 简历结构化拆分：按教育经历、项目经历、工作经历、技能等 section 拆分
- 向量检索：使用 DashScope `text-embedding-v4` 写入 Chroma
- 多轮问答：按 `user_id` 维护最近 5 轮对话记忆
- HR 简历助手 Prompt：只基于检索上下文回答，避免编造
- 来源追溯：返回命中文档片段、页码、文件名和相似度分数
- 文档管理：支持查看用户已上传文档和删除指定文档

## 项目结构

```text
ai-rag-demo/
├── conversations_rag.py       # FastAPI 主服务：上传、问答、流式输出、文档管理
├── main.py                    # DeepSeek Chat API 最小调用示例
├── rag_demo.py                # 单 PDF 入库和基础检索示例
├── test_fastapi.py            # FastAPI 最小接口测试示例
├── requirements.txt           # Python 依赖
├── README.md
├── chroma_db/                 # Chroma 本地向量数据库目录
├── docs/                      # 上传文档保存目录
├── prompts/
│   └── hr_prompt.py           # HR 简历问答 Prompt
├── splitter/
│   ├── chunk_splitter.py      # 通用 chunk 切分逻辑
│   └── resume_splitter.py     # 简历 section 结构化拆分逻辑
└── utils/
    ├── metadatas.py           # metadata 构建与校验
    └── text_cleaner.py        # 文本标准化工具
```

## 环境准备

建议使用 Python 3.10 或 3.11。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=你的通义千问/DashScope API Key
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## 启动服务

```bash
python conversations_rag.py
```

服务默认运行在：

```text
http://localhost:8000
```

FastAPI 文档地址：

```text
http://localhost:8000/docs
```

## 常用接口

### 上传文档

`POST /upload`

表单参数：

- `user_id`：用户 ID
- `file`：上传文件

返回示例：

```json
{
  "message": "上传成功",
  "filename": "resume.pdf",
  "chunks": 12
}
```

### 普通问答

`POST /ask`

```json
{
  "user_id": "user-001",
  "question": "候选人有哪些项目经历？"
}
```

### 流式问答

`POST /chat_stream`

请求体同 `/ask`，返回 `text/plain` 流式文本。

### 查看检索来源和历史

`POST /sources_history`

```json
{
  "user_id": "user-001",
  "question": "候选人掌握哪些技能？"
}
```

### 查看已上传文档

`GET /documents?user_id=user-001`

### 删除文档

`DELETE /delete_document?user_id=user-001&file_hash=xxx`

## 处理流程

```text
上传文件
  ↓
保存到 docs/
  ↓
解析文档内容
  ↓
按简历 section 拆分
  ↓
按 chunk_size 继续切片
  ↓
构建标准 metadata
  ↓
写入 Chroma 向量库
  ↓
用户提问时按 user_id 检索相关 chunk
  ↓
结合 HR_PROMPT 生成回答
```

## 注意事项

- `chroma_db/` 是本地向量库目录，删除后需要重新上传文档入库。
- `docs/` 保存上传的原始文件，删除文档接口会同步删除对应文件。
- 当前会话记忆保存在进程内存中，服务重启后会丢失。
- `/ask` 和 `/chat_stream` 会按 `user_id` 过滤检索结果，避免不同用户的文档互相串用。
