# 处理聊天相关的路由
from fastapi import APIRouter

from services.rag_service import llm, vectorstore
from retrievers.custom_retriever import CustomRerankRetriever
from schemas.chat_schema import UserQuery
from config.rag_config import (
    RECALL_K,
    RERANK_TOP_K,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    COMPRESS_CONTEXT,
)

router = APIRouter()


@router.post("/retrieve_evidence")
def retrieve_evidence(query: UserQuery):
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

    return {
        "evidence": [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ]
    }
