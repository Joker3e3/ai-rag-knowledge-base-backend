# BM25 内存池管理模块

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

bm25_docs = []

def rebuild_bm25_docs_from_chroma(vectorstore):
    """
    从 Chroma 中读取所有已保存的 documents，
    重新构建 BM25 所需的内存 Document 列表。
    """
    global bm25_docs

    data = vectorstore.get()

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    rebuilt_docs = []

    for content, metadata in zip(documents, metadatas):
        if not content:
            continue

        rebuilt_docs.append(
            Document(
                page_content=content,
                metadata=metadata or {}
            )
        )

    bm25_docs = rebuilt_docs

    return bm25_docs


def add_docs_to_bm25(docs):
    """
    上传新文件后，把新 chunk 同步加入 BM25 内存池。
    """
    bm25_docs.extend(docs)


def get_user_bm25_retriever(user_id, k=3):
    """
    根据 user_id 构建当前用户专属 BM25Retriever。
    """
    user_docs = [
        doc for doc in bm25_docs
        if doc.metadata.get("user_id") == user_id
    ]

    if not user_docs:
        return None

    retriever = BM25Retriever.from_documents(user_docs)
    retriever.k = k

    return retriever

def get_resume_bm25_retriever(user_id, candidate_id, resume_id, k=3):
    """
    根据 candidate_id + resume_id 构建 BM25Retriever
    """

    resume_docs = [
        doc for doc in bm25_docs
        if doc.metadata.get("user_id") == user_id
        and doc.metadata.get("candidate_id") == candidate_id
        and doc.metadata.get("resume_id") == resume_id
    ]

    if not resume_docs:
        return None

    retriever = BM25Retriever.from_documents(resume_docs)
    retriever.k = k

    return retriever

def remove_docs_from_bm25(user_id, file_hash):
    """
    删除文件后，把对应的 chunk 从 BM25 内存池中移除。
     - user_id 和 file_hash 共同定位到某个文件的所有 chunk。
    """
    global bm25_docs

    bm25_docs = [
        doc for doc in bm25_docs
        if not (
            doc.metadata.get("user_id") == user_id
            and doc.metadata.get("file_hash") == file_hash
        )
    ]