import os
import uuid
import hashlib

from fastapi import UploadFile, HTTPException, BackgroundTasks
from langchain.schema import Document

from services.rag_service import vectorstore
from services.redis_service import redis_client
from loaders.document_loader import load_document
from splitters.resume_splitter import split_resume_sections
from splitters.chunk_splitter import split_chunks
from utils.metadatas import build_metadata
from retrievers.bm25_store import add_docs_to_bm25, remove_docs_from_bm25
from config.rag_config import DOCS_DIR

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".csv", ".xlsx", ".xls"}

document_status_store = {}


async def upload_document(
    background_tasks: BackgroundTasks, user_id: str, file: UploadFile
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")

    max_file_size = 10 * 1024 * 1024

    if len(content) > max_file_size:
        raise HTTPException(status_code=400, detail="文件不能超过10MB")

    file_hash = hashlib.md5(content).hexdigest()

    existing = vectorstore.get(
        where={"$and": [{"file_hash": file_hash}, {"user_id": user_id}]}
    )

    if existing["ids"]:
        raise HTTPException(status_code=400, detail="该文件已上传")

    os.makedirs(DOCS_DIR, exist_ok=True)

    unique_name = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(DOCS_DIR, unique_name)

    with open(save_path, "wb") as f:
        f.write(content)

    # 给待处理文件加入状态
    set_document_status(
        file_hash=file_hash,
        filename=file.filename,
        user_id=user_id,
        status="processing",
    )
    # document_status_store[file_hash] = {
    #     "filename": file.filename,
    #     "file_hash": file_hash,
    #     "user_id": user_id,
    #     "status": "processing",
    # }

    background_tasks.add_task(
        process_document, save_path, user_id, file.filename, unique_name, file_hash
    )

    return {
        "message": "文件上传成功，正在处理中",
        "status": "processing",
        "filename": file.filename,
    }


def process_document(save_path, user_id, filename, unique_name, file_hash):
    try:
        docs = load_document(save_path)

        for doc in docs:
            doc.metadata["filename"] = filename
            doc.metadata["saved_filename"] = unique_name
            doc.metadata["user_id"] = user_id
            doc.metadata["file_hash"] = file_hash

        section_docs = split_resume_sections(docs)
        split_docs = split_chunks(section_docs)

        new_docs = []

        for index, chunk in enumerate(split_docs):

            metadata = build_metadata(
                user_id=chunk.metadata["user_id"],
                filename=chunk.metadata["filename"],
                saved_filename=chunk.metadata["saved_filename"],
                file_hash=chunk.metadata["file_hash"],
                page=chunk.metadata.get("page", 0),
                section=chunk.metadata.get("section", "其他"),
                parent_id=chunk.metadata.get("parent_id"),
                chunk_index=index,
                document_status="ready",
            )

            new_doc = Document(page_content=chunk.page_content, metadata=metadata)

            new_docs.append(new_doc)

        vectorstore.add_documents(new_docs)
        add_docs_to_bm25(new_docs)

        # document_status_store[file_hash]["status"] = "ready"
        update_document_status(file_hash=file_hash, status="ready")
        print(f"{filename} 文档处理完成")

    except Exception as e:

        update_document_status(file_hash=file_hash, status="failed", error=str(e))
        # document_status_store[file_hash]["status"] = "failed"
        # document_status_store[file_hash]["error"] = str(e)

        print(f"{filename} 文档处理失败: {str(e)}")


def list_documents(user_id: str):
    docs = vectorstore.get(where={"user_id": user_id})

    metadatas = docs.get("metadatas", [])

    unique_files = {}

    for metadata in metadatas:
        file_hash = metadata.get("file_hash")

        if not file_hash:
            continue

        if file_hash not in unique_files:
            unique_files[file_hash] = {
                "filename": metadata.get("filename"),
                "file_hash": file_hash,
                "status": "ready",
            }

    # for file_hash, info in document_status_store.items():
    #     if info.get("user_id") != user_id:
    #         continue

    #     unique_files[file_hash] = {
    #         "filename": info.get("filename"),
    #         "file_hash": file_hash,
    #         "status": info.get("status"),
    #         "error": info.get("error"),
    #     }

    all_status = get_all_document_status()

    for info in all_status:

        if info.get("user_id") != user_id:
            continue

        file_hash = info.get("file_hash")

        unique_files[file_hash] = {
            "filename": info.get("filename"),
            "file_hash": file_hash,
            "status": info.get("status"),
            "error": info.get("error"),
        }

    return list(unique_files.values())


def delete_document_by_hash(user_id: str, file_hash: str):
    docs = vectorstore.get(
        where={"$and": [{"file_hash": file_hash}, {"user_id": user_id}]}
    )

    if not docs["ids"]:
        raise HTTPException(status_code=404, detail="文档不存在")

    saved_filenames = set()

    for metadata in docs["metadatas"]:
        saved_filename = metadata.get("saved_filename")

        if saved_filename:
            saved_filenames.add(saved_filename)

    vectorstore.delete(where={"$and": [{"user_id": user_id}, {"file_hash": file_hash}]})

    remove_docs_from_bm25(user_id, file_hash)
    delete_document_status(file_hash)
    # document_status_store.pop(file_hash, "")

    for saved_filename in saved_filenames:
        file_path = os.path.join(DOCS_DIR, saved_filename)

        if os.path.exists(file_path):
            os.remove(file_path)

    return {"message": "文档已删除"}


"""
Redis文档状态管理
"""


def set_document_status(file_hash, filename, user_id, status, error=None):
    redis_client.hset(
        f"document:{file_hash}",
        mapping={
            "file_hash": file_hash,
            "filename": filename,
            "user_id": user_id,
            "status": status,
            "error": error or "",
        },
    )


def get_document_status(file_hash):
    return redis_client.hgetall(f"document:{file_hash}")


def update_document_status(file_hash, status, error=None):
    mapping = {"status": status}

    if error:
        mapping["error"] = error

    redis_client.hset(f"document:{file_hash}", mapping=mapping)


def delete_document_status(file_hash):
    redis_client.delete(f"document:{file_hash}")


def get_all_document_status():

    keys = redis_client.keys("document:*")

    results = []

    for key in keys:

        data = redis_client.hgetall(key)

        if data:
            results.append(data)

    return results
