import logging
import uuid

from fastapi import APIRouter

from config.rag_config import (
    BM25_WEIGHT,
    COMPRESS_CONTEXT,
    RECALL_K,
    RERANK_TOP_K,
    VECTOR_WEIGHT,
)
from retrievers.custom_retriever import CustomRerankRetriever
from schemas.chat_schema import UserQuery
from services.rag_service import llm, vectorstore
from utils.rag_timing import log_rag_timing, new_timer


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/retrieve_evidence")
def retrieve_evidence(query: UserQuery):
    request_id = uuid.uuid4().hex[:8]
    total_start = new_timer()

    retriever = CustomRerankRetriever(
        vectorstore=vectorstore,
        user_id=query.user_id,
        llm=llm,
        recall_k=RECALL_K,
        rerank_top_k=RERANK_TOP_K,
        bm25_weight=BM25_WEIGHT,
        vector_weight=VECTOR_WEIGHT,
        compress_context=COMPRESS_CONTEXT,
        request_id=request_id,
    )

    try:
        docs = retriever.invoke(query.question)
    finally:
        log_rag_timing(logger, request_id, "total", total_start)

    return {
        "evidence": [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ],
    }
