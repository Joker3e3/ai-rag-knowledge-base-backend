# conversations_rag.py
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from fastapi.responses import StreamingResponse
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
from dotenv import load_dotenv
import asyncio
import uuid
import hashlib

from utils.metadatas import build_metadata, validate_metadata
from prompts.hr_prompt import HR_PROMPT
from splitter.resume_splitter import split_resume_sections
from splitter.chunk_splitter import split_chunks
from retrievers.bm25_store import (
    rebuild_bm25_docs_from_chroma,
    add_docs_to_bm25,
    get_user_bm25_retriever,
    remove_docs_from_bm25,
)
from retrievers.reranker import rerank_documents
from retrievers.custom_retriever import CustomRerankRetriever
from config.rag_config import (
    RECALL_K,
    RERANK_TOP_K,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    COMPRESS_CONTEXT,
)

load_dotenv()
os.makedirs("./docs", exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}

# -----------------------
# 2️⃣ Embedding + Vectorstore
# -----------------------
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# -----------------------
# 重建全局 BM25 文档池
# -----------------------
bm25_docs = rebuild_bm25_docs_from_chroma(vectorstore)


def load_document(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)

    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")

    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)

    elif ext == ".csv":
        loader = CSVLoader(file_path)

    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    return loader.load()


# -----------------------
# 3️⃣ Memory 初始化
# -----------------------
"""
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)
"""
# Memory 会自动保存每轮问答，方便多轮对话

# -----------------------
# 用户 Memory 存储
# -----------------------

# 用 dict 保存：
# key = user_id
# value = ConversationBufferMemory

memory_store = {}


# 获取当前用户的 Memory
def get_user_memory(user_id):

    if user_id not in memory_store:

        # windowMemory 只保存最近 k 轮对话
        memory_store[user_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
            k=5,  # 保存最近5轮对话
        )

    return memory_store[user_id]


# -----------------------
# 4️⃣ 创建 RAG Chain
# -----------------------
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
    # 控制随机性：0=最严谨，1=最有创意
    temperature=0.3,
    streaming=True,
    # 超时时间，避免卡死
    timeout=60.0,
)
# RetrievalQA

# LangChain 提供的“问答链”
""" 会自动做两件事：
# 1️⃣ 用 Retriever 找到最相似的文本块
# 2️⃣ 把这些文本块拼接进 LLM Prompt，让 LLM 生成答案
# 作用：把“检索 + 生成”整合成一个调用

只能做 单轮检索 + 回答
"""
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True
# )
"""Chain != 函数

Chain 是：

“工作流”
"""
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     memory=memory,
#     return_source_documents=True
# )

MAX_DISTANCE = 1.2  # 向量距离阈值

# -----------------------
# 5️⃣ FastAPI 初始化
# -----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # 允许哪些前端地址访问
    allow_origins=["http://localhost:5173"],
    # 允许携带 cookie
    allow_credentials=True,
    # 允许所有 HTTP 方法
    allow_methods=["*"],
    # 允许所有请求头
    allow_headers=["*"],
)


# 数据校验模型
# FastAPI 自动要求：请求 JSON
class UserQuery(BaseModel):
    user_id: str
    question: str


class ClearUserMemory(BaseModel):
    user_id: str


# 装饰器
# 路由：当收到 POST /ask 请求时，会调用下方函数
@app.post("/ask")
async def ask(query: UserQuery):
    # 包装输入，加上 Memory 历史
    # history = memory.load_memory_variables({})  # 返回字典
    # chat_history_str = history.get("chat_history", "")  # 取出你定义的 memory_key

    memory = get_user_memory(query.user_id)

    docs_and_scores = vectorstore.similarity_search_with_score(query.question, k=3)
    best_score = docs_and_scores[0][1]  # 取出最相似块的距离分数

    # if best_score > MAX_DISTANCE:
    #     return {
    #         "content": "未在文档中找到相关内容。",
    #         "chat_history": memory.load_memory_variables({}),
    #         "sources": docs_and_scores
    #     }

    # 把向量库包装成“检索器（Retriever）”
    # 作用：给 LLM 提供语义检索接口
    ## mmr避免重复chunk
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "filter": {"user_id": query.user_id}},
    )
    # 返回 top3 相似块
    # k=3 表示每次查询返回 最相似的 3 个文本块
    # 为什么要返回多个？
    # 避免单个块信息不足
    # 提高回答准确率
    # 返回的每个块都是之前 PDF 切片 + embedding 的文本

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": HR_PROMPT},
    )
    # 调用 RAG
    # Retriever 粗召回 LLM 精判断
    result = qa_chain.invoke({"question": query.question})
    answer = result["answer"] if isinstance(result, dict) else result

    # 解析来源信息
    source_docs = result["source_documents"]
    sources = []

    for doc in source_docs:
        sources.append({"content": doc.page_content, "metadata": doc.metadata})

    # Memory 更新
    """
    实际上这里并不需要手动存取历史记录，因为ConversationalRetrievalChain在invoke时会自动读写并更新memory
    """
    # memory.save_context({"question": query.question}, {"answer": answer})

    # 返回给前端JSON
    return {
        "content": answer,
        "chat_history": memory.load_memory_variables({}).get(
            "chat_history", ""
        ),  # 返回最近5轮
        "sources": sources,
    }


@app.post("/clear_memory")
async def clear_memory(request: ClearUserMemory):
    user_id = request.user_id

    if user_id in memory_store:
        del memory_store[user_id]
        return {"message": f"Memory for user {user_id} cleared."}
    else:
        return {"message": f"No memory found for user {user_id}."}


@app.get("/llm_stream")
async def llm_stream():

    async def generate():

        # 真正调用 LLM stream
        for chunk in llm.stream("请简单介绍一下人工智能"):

            # chunk 是 AI 返回的小块 token
            content = chunk.content

            if content:

                print(content, end="", flush=True)

                yield content

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/chat_stream")
async def chat_stream(query: UserQuery):

    memory = get_user_memory(query.user_id)

    retriever = CustomRerankRetriever(
        vectorstore=vectorstore,
        user_id=query.user_id,
        bm25_weight=BM25_WEIGHT,
        vector_weight=VECTOR_WEIGHT,
        recall_k=RECALL_K,
        rerank_top_k=RERANK_TOP_K,
        llm=llm,
        compress_context=COMPRESS_CONTEXT,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": HR_PROMPT},
    )

    async def generate():

        async for chunk in qa_chain.astream({"question": query.question}):
            print(chunk)
            if "answer" in chunk:
                content = chunk["answer"]
                if content:
                    yield content

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/sources_history")
async def sources_history(query: UserQuery):
    memory = get_user_memory(query.user_id)
    history = memory.load_memory_variables({})

    sources = []

    # 初版向量化搜索版本，弃用，改用 CustomRerankRetriever 里的混合搜索+rerank后的压缩版本
    # docs = vectorstore.similarity_search_with_score(
    #     query.question, k=3, filter={"user_id": query.user_id}
    # )
    retriever = CustomRerankRetriever(
        vectorstore=vectorstore,
        user_id=query.user_id,
        bm25_weight=BM25_WEIGHT,
        vector_weight=VECTOR_WEIGHT,
        recall_k=RECALL_K,
        rerank_top_k=RERANK_TOP_K,
        llm=llm,
        compress_context=COMPRESS_CONTEXT,
    )

    docs = retriever.invoke(query.question)

    for doc in docs:
        sources.append(
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "content": doc.page_content[:200],
                "filename": doc.metadata.get("filename"),
                "user_id": doc.metadata.get("user_id"),
                "score": f"{doc.metadata.get('rerank_score'):.2f}",
            }
        )

    return {"chat_history": history.get("chat_history", ""), "sources": sources}


@app.post("/upload")
async def upload(user_id: str = Form(...), file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    unique_name = f"{uuid.uuid4()}{ext}"

    # 保存路径
    save_path = os.path.join("./docs", unique_name)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")

    MAX_FILE_SIZE = 10 * 1024 * 1024
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件不能超过10MB")

    file_hash = hashlib.md5(content).hexdigest()

    # 查询是否已存在
    existing = vectorstore.get(
        where={"$and": [{"file_hash": file_hash}, {"user_id": user_id}]}
    )
    if existing["ids"]:
        raise HTTPException(status_code=400, detail="该文件已上传")

    # 保存文件
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        docs = load_document(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档解析失败: {str(e)}")

    # 文档级metadata
    for doc in docs:
        doc.metadata["filename"] = file.filename
        doc.metadata["saved_filename"] = unique_name
        doc.metadata["user_id"] = user_id
        doc.metadata["file_hash"] = file_hash

    # 进行按简历结构拆分
    section_docs = split_resume_sections(docs)

    # 按照 chunk_size 进行切分
    split_docs = split_chunks(section_docs)

    # 最终 metadata 标准化
    new_docs = []
    for index, chunk in enumerate(split_docs):
        metadata = build_metadata(
            user_id=chunk.metadata["user_id"],
            filename=chunk.metadata["filename"],
            saved_filename=chunk.metadata["saved_filename"],
            file_hash=chunk.metadata["file_hash"],
            parent_id=chunk.metadata.get("parent_id"),
            page=chunk.metadata.get("page", 0),
            section=chunk.metadata.get("section", "其他"),
            chunk_index=index,
        )
        chunk.metadata = metadata
        new_docs.append(chunk)

    # 先持久化存储
    vectorstore.add_documents(new_docs)

    # 再维护内存级 BM25 索引
    add_docs_to_bm25(new_docs)

    return {"message": "上传成功", "filename": file.filename, "chunks": len(split_docs)}


@app.get("/documents")
async def list_documents(user_id: str):

    docs = vectorstore.get(where={"user_id": user_id})
    unique_files = {}
    metadatas = docs["metadatas"]

    for metadata in metadatas:

        file_hash = metadata["file_hash"]
        if file_hash not in unique_files:
            unique_files[file_hash] = {
                "filename": metadata["filename"],
                "file_hash": file_hash,
            }

    return list(unique_files.values())


@app.delete("/delete_document")
async def delete_document(user_id: str, file_hash: str):

    docs = vectorstore.get(
        where={"$and": [{"file_hash": file_hash}, {"user_id": user_id}]}
    )

    if not docs["ids"]:
        raise HTTPException(status_code=404, detail="文档不存在")

    metadatas = docs["metadatas"]

    saved_filenames = set()

    for metadata in metadatas:
        saved_filename = metadata.get("saved_filename")
        if saved_filename:
            saved_filenames.add(saved_filename)

    vectorstore.delete(where={"$and": [{"user_id": user_id}, {"file_hash": file_hash}]})

    remove_docs_from_bm25(user_id=user_id, file_hash=file_hash)

    for saved_filename in saved_filenames:
        file_path = os.path.join("./docs", saved_filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    return {"message": "文档已删除"}


@app.post("/debug_rerank")
async def debug_rerank(query: UserQuery):

    docs_and_scores = vectorstore.similarity_search_with_score(query.question, k=10)
    docs = [doc for doc, score in docs_and_scores]

    reranked_docs = rerank_documents(query.question, docs, top_k=3)

    results = []
    for doc, score in reranked_docs:
        results.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "rerank_score": float(score),
            }
        )

    return {"reranked_results": results}


# 启动 FastAPI 应用
if __name__ == "__main__":
    import uvicorn

    # uvicorn.run 启动开发服务器
    # 参数：
    # "conversations_rag:app" → conversations_rag.py 文件中的 app 实例
    # host="0.0.0.0" → 允许外部访问
    # port=8000 → 监听端口
    # reload=True → 文件修改时自动重启（开发模式用）
    uvicorn.run("conversations_rag:app", host="0.0.0.0", port=8000, reload=True)
