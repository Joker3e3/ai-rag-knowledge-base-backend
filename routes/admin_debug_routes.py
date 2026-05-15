# debug 路由
from fastapi import APIRouter
from langchain.chains import ConversationalRetrievalChain

from fastapi.responses import StreamingResponse


from prompts.hr_prompt import HR_PROMPT
from retrievers.reranker import rerank_documents
from schemas.chat_schema import UserQuery
from services.rag_service import llm, vectorstore
from services.memory_service import get_user_memory

router = APIRouter()


# 装饰器
# 路由：当收到 POST /ask 请求时，会调用下方函数
@router.post("/ask")
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


@router.get("/llm_stream")
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


@router.post("/debug_rerank")
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
