from typing import List
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain.retrievers import EnsembleRetriever

from retrievers.bm25_store import get_user_bm25_retriever
from retrievers.reranker import rerank_documents
from retrievers.context_compressor import compress_documents


# 使用向量检索和BM25检索的混合方法进行检索，并对结果进行重新排序
class CustomRerankRetriever(BaseRetriever):

    vectorstore: Any
    user_id: str
    bm25_weight: float
    vector_weight: float

    recall_k: int
    rerank_top_k: int

    llm: Any

    compress_context: bool = False

    def _get_relevant_documents(
        self,
        query: str
    ) -> List[Document]:

        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": self.recall_k,
                "filter": {
                    "user_id": self.user_id
                }
            }
        )

        bm25_retriever = get_user_bm25_retriever(
            self.user_id,
            k=self.recall_k
        )

        if bm25_retriever:

            hybrid_retriever = EnsembleRetriever(
                retrievers=[
                    bm25_retriever,
                    vector_retriever
                ],
                weights=[self.bm25_weight, self.vector_weight]
            )

            docs = hybrid_retriever.invoke(query)

        else:
            docs = vector_retriever.invoke(query)

        # Rerank后的文档列表，包含文档和对应的分数
        reranked_docs = rerank_documents(
            query,
            docs,
            top_k=self.rerank_top_k
        )

        final_docs = []

        for doc, score in reranked_docs:

            doc.metadata["rerank_score"] = float(score)
            final_docs.append(doc)

        if self.compress_context:

            # 用LLM对Rerank后的文档进行压缩，提取与用户问题最相关的内容，减少上下文长度
            compressed_docs = compress_documents(
                llm=self.llm,
                question=query,
                docs=final_docs
            )
        
        else:
            compressed_docs = final_docs

        return compressed_docs