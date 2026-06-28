import logging
import uuid
from typing import Any, List

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config.rag_config import RERANK_ENABLED
from retrievers.bm25_store import get_resume_bm25_retriever, get_user_bm25_retriever
from retrievers.context_compressor import compress_documents
from retrievers.query_rewriter import rewrite_search_query
from retrievers.reranker import rerank_documents
from utils.rag_timing import log_rag_timing, new_timer


logger = logging.getLogger(__name__)


class CustomRerankRetriever(BaseRetriever):
    vectorstore: Any
    user_id: str
    candidate_id: str | None = None
    resume_id: str | None = None

    bm25_weight: float
    vector_weight: float

    recall_k: int
    rerank_top_k: int
    rerank_enabled: bool = RERANK_ENABLED

    llm: Any

    compress_context: bool = False
    rewrite_query: str | None = None
    request_id: str | None = None

    def _get_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:
        request_id = self.request_id or uuid.uuid4().hex[:8]

        search_query = rewrite_search_query(self.llm, query, self.rewrite_query)

        if self.candidate_id and self.resume_id:
            filters = {
                "$and": [
                    {"user_id": self.user_id},
                    {"candidate_id": self.candidate_id},
                    {"resume_id": self.resume_id},
                ]
            }
        else:
            filters = {"user_id": self.user_id}
            
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": self.recall_k,
                "filter": filters,
            },
        )

        bm25_start = new_timer()
        bm25_docs = []
        bm25_retriever = None
        try:
            if self.candidate_id and self.resume_id:
                bm25_retriever = get_resume_bm25_retriever(
                    self.user_id,
                    self.candidate_id,
                    self.resume_id,
                    k=self.recall_k,
                )
            else:
                bm25_retriever = get_user_bm25_retriever(
                    self.user_id,
                    k=self.recall_k,
                )

            if bm25_retriever:
                bm25_docs = bm25_retriever.invoke(search_query)
        finally:
            log_rag_timing(logger, request_id, "bm25", bm25_start)

        vector_start = new_timer()
        try:
            vector_docs = vector_retriever.invoke(search_query)
        finally:
            log_rag_timing(logger, request_id, "vector", vector_start)

        merge_start = new_timer()
        try:
            if bm25_retriever:
                hybrid_retriever = EnsembleRetriever(
                    retrievers=[
                        bm25_retriever,
                        vector_retriever,
                    ],
                    weights=[self.bm25_weight, self.vector_weight],
                )

                docs = hybrid_retriever.weighted_reciprocal_rank(
                    [
                        bm25_docs,
                        vector_docs,
                    ],
                )
            else:
                docs = vector_docs
        finally:
            log_rag_timing(logger, request_id, "hybrid_merge", merge_start)

        rerank_start = new_timer()
        if self.rerank_enabled:
            try:
                reranked_docs = rerank_documents(
                    query,
                    docs,
                    top_k=self.rerank_top_k,
                )
            finally:
                log_rag_timing(logger, request_id, "rerank", rerank_start)

            final_docs = []

            for doc, score in reranked_docs:
                doc.metadata["rerank_score"] = float(score)
                final_docs.append(doc)
        else:
            # Benchmark mode: keep BM25 + vector + hybrid merge, but skip the
            # CPU reranker to compare latency and evidence quality directly.
            final_docs = docs[: self.rerank_top_k]
            for doc in final_docs:
                doc.metadata["rerank_score"] = None
                doc.metadata["rerank_enabled"] = False

            log_rag_timing(
                logger,
                request_id,
                "rerank_skipped",
                duration_ms_value=0.0,
                reason="RERANK_ENABLED_FALSE",
            )

        logger.info(
            (
                "[RAG_REQUEST] request_id=%s rerank_enabled=%s recall_k=%s "
                "rerank_top_k=%s candidate_count_before_rerank=%s "
                "final_result_count=%s"
            ),
            request_id,
            self.rerank_enabled,
            self.recall_k,
            self.rerank_top_k,
            len(docs),
            len(final_docs),
        )

        if self.compress_context:
            compressed_docs = compress_documents(
                llm=self.llm,
                question=query,
                docs=final_docs,
            )
        else:
            compressed_docs = final_docs

        return compressed_docs
