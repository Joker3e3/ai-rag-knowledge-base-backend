import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain.chains import ConversationalRetrievalChain

from config.rag_config import (
    BM25_WEIGHT,
    COMPRESS_CONTEXT,
    RECALL_K,
    RERANK_TOP_K,
    VECTOR_WEIGHT,
)
from prompts.hr_prompt import HR_PROMPT
from retrievers.custom_retriever import CustomRerankRetriever
from schemas.chat_schema import UserQuery
from services.memory_service import get_user_memory
from services.rag_service import llm, vectorstore
from utils.rag_timing import log_rag_timing, new_timer


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat_stream")
async def chat_stream(query: UserQuery):
    request_id = uuid.uuid4().hex[:8]
    total_start = new_timer()
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
        request_id=request_id,
    )

    prompt_start = new_timer()
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": HR_PROMPT},
        )
    finally:
        log_rag_timing(logger, request_id, "prompt_assembly", prompt_start)

    async def generate():
        llm_start = new_timer()
        try:
            async for chunk in qa_chain.astream({"question": query.question}):
                if "answer" in chunk:
                    content = chunk["answer"]
                    if content:
                        yield content
        finally:
            log_rag_timing(logger, request_id, "llm_stream_total", llm_start)
            log_rag_timing(logger, request_id, "total", total_start)

    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/sources_history")
async def sources_history(query: UserQuery):
    request_id = uuid.uuid4().hex[:8]
    total_start = new_timer()
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
        request_id=request_id,
    )

    try:
        docs = retriever.invoke(query.question)
    finally:
        log_rag_timing(logger, request_id, "total", total_start)

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
            },
        )

    return {"chat_history": history.get("chat_history", ""), "sources": sources}
