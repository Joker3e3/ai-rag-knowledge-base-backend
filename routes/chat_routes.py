# 处理聊天相关的路由
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.chains import ConversationalRetrievalChain
from services.rag_service import llm, vectorstore
from services.memory_service import get_user_memory
from retrievers.custom_retriever import CustomRerankRetriever
from prompts.hr_prompt import HR_PROMPT
from schemas.chat_schema import UserQuery
from config.rag_config import (
    RECALL_K,
    RERANK_TOP_K,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    COMPRESS_CONTEXT,
)

router = APIRouter()

@router.post("/chat_stream")
async def chat_stream(query: UserQuery):
    memory = get_user_memory(query.user_id)

    retriever = CustomRerankRetriever(
        vectorstore=vectorstore,
        user_id=query.user_id,
        llm=llm,
        recall_k=RECALL_K,
        rerank_top_k=RERANK_TOP_K,
        bm25_weight=BM25_WEIGHT,
        vector_weight=VECTOR_WEIGHT,
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
            if "answer" in chunk:
                content = chunk["answer"]
                if content:
                    yield content

    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/sources_history")
async def sources_history(query: UserQuery):
    memory = get_user_memory(query.user_id)
    history = memory.load_memory_variables({})

    retriever = CustomRerankRetriever(
        vectorstore=vectorstore,
        user_id=query.user_id,
        llm=llm,
        recall_k=RECALL_K,
        rerank_top_k=RERANK_TOP_K,
        bm25_weight=BM25_WEIGHT,
        vector_weight=VECTOR_WEIGHT,
        compress_context=COMPRESS_CONTEXT,
    )

    docs = retriever.invoke(query.question)

    sources = []

    for doc in docs:
        sources.append(
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "content": doc.page_content[:300],
                "filename": doc.metadata.get("filename"),
                "user_id": doc.metadata.get("user_id"),
                "section": doc.metadata.get("section"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "parent_id": doc.metadata.get("parent_id"),
                "rerank_score": doc.metadata.get("rerank_score"),
            }
        )

    return {"chat_history": history.get("chat_history", ""), "sources": sources}
