import json
import os
import re
import uuid
import hashlib

from fastapi import UploadFile, HTTPException, BackgroundTasks
from langchain.schema import Document

from database.database import SessionLocal
from database.models.candidate import Candidate
from database.models.resume import Resume
from services.rag_service import vectorstore
from services.redis_service import redis_client
from loaders.document_loader import load_document
from splitters.resume_splitter import split_resume_sections
from splitters.chunk_splitter import split_chunks
from utils.metadatas import build_metadata
from retrievers.bm25_store import add_docs_to_bm25, remove_docs_from_bm25
from config.rag_config import DOCS_DIR

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".csv", ".xlsx", ".xls"}
VECTORSTORE_ADD_BATCH_SIZE = 10

document_status_store = {}

def parse_candidate_info_from_filename(filename: str) -> dict:
    """
    通过解析文件名获取候选人信息，要求文件名格式为：姓名_学校_手机号.pdf
    例如：张三_山东大学_13812345678.pdf
    """
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split("_")

    if len(parts) != 3:
        raise HTTPException(
            status_code=400,
            detail="文件名格式错误，请使用：姓名_学校_手机号.pdf，例如：张三_山东大学_13812345678.pdf",
        )

    candidate_name = parts[0].strip()
    school = parts[1].strip()
    phone = parts[2].strip()

    if not candidate_name or not school or not phone:
        raise HTTPException(
            status_code=400,
            detail="文件名中的姓名、学校、手机号不能为空",
        )

    if not re.fullmatch(r"1[3-9]\d{9}", phone):
        raise HTTPException(
            status_code=400,
            detail="手机号格式错误，请使用11位中国大陆手机号",
        )

    return {
        "candidate_name": candidate_name,
        "school": school,
        "phone": phone,
    }

async def upload_document(
    background_tasks: BackgroundTasks, user_id: str, file: UploadFile
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    candidate_info = parse_candidate_info_from_filename(file.filename)

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
        saved_filename=unique_name,
    )
    # document_status_store[file_hash] = {
    #     "filename": file.filename,
    #     "file_hash": file_hash,
    #     "user_id": user_id,
    #     "status": "processing",
    # }

    background_tasks.add_task(
        process_document, save_path, user_id, file.filename, unique_name, file_hash, candidate_info
    )

    return {
        "message": "文件上传成功，正在处理中",
        "status": "processing",
        "filename": file.filename,
    }


def process_document(save_path, user_id, filename, unique_name, file_hash, candidate_info):
    try:
        docs = load_document(save_path)

        candidate_name = candidate_info["candidate_name"]
        school = candidate_info["school"]
        phone = candidate_info["phone"]

        candidate_identity = f"{user_id}:{candidate_name}:{school}:{phone}"
        candidate_id = hashlib.md5(candidate_identity.encode("utf-8")).hexdigest()

        resume_id = str(uuid.uuid4())

        for doc in docs:
            doc.metadata["filename"] = filename
            doc.metadata["saved_filename"] = unique_name
            doc.metadata["user_id"] = user_id
            doc.metadata["file_hash"] = file_hash
            doc.metadata["candidate_id"] = candidate_id
            doc.metadata["resume_id"] = resume_id
            doc.metadata["candidate_name"] = candidate_name
            doc.metadata["school"] = school
            doc.metadata["phone"] = phone
            doc.metadata["is_latest"] = True

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

                candidate_id=chunk.metadata["candidate_id"],
                resume_id=chunk.metadata["resume_id"],
            )

            new_doc = Document(page_content=chunk.page_content, metadata=metadata)

            new_docs.append(new_doc)

        db = SessionLocal()

        try:
            existing = db.query(Candidate).filter(
                Candidate.candidate_id == candidate_id
            ).first()

            if not existing:
                candidate = Candidate(
                    candidate_id=candidate_id,
                    user_id=user_id,
                    name=candidate_name,
                    school=school,
                    phone=phone
                )

                db.add(candidate)

            db.query(Resume).filter(
                Resume.candidate_id == candidate_id
            ).update({"is_latest": False})
            resume = Resume(
                resume_id=resume_id,
                candidate_id=candidate_id,
                file_name=filename,
                file_hash=file_hash,
                is_latest=True
            )

            db.add(resume)
            db.commit()

        finally:
            db.close()

        add_documents_to_vectorstore(new_docs)

        add_docs_to_bm25(new_docs)

        # document_status_store[file_hash]["status"] = "ready"
        update_document_status(file_hash=file_hash, user_id=user_id, status="ready")
        print(f"{filename} 文档处理完成")

    except Exception as e:

        cleanup_document_indexes(user_id, file_hash)
        update_document_status(file_hash=file_hash, user_id=user_id, status="failed", error=str(e))
        # document_status_store[file_hash]["status"] = "failed"
        # document_status_store[file_hash]["error"] = str(e)

        print(f"{filename} 文档处理失败: {str(e)}")


def add_documents_to_vectorstore(docs):
    for start in range(0, len(docs), VECTORSTORE_ADD_BATCH_SIZE):
        batch = docs[start:start + VECTORSTORE_ADD_BATCH_SIZE]
        vectorstore.add_documents(batch)


def cleanup_document_indexes(user_id, file_hash):
    try:
        vectorstore.delete(where={"$and": [{"file_hash": file_hash}, {"user_id": user_id}]})
    except Exception as e:
        print(f"cleanup chroma failed for {file_hash}: {str(e)}")

    remove_docs_from_bm25(user_id, file_hash)


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

    if docs["ids"]:
        saved_filenames = set()

        for metadata in docs["metadatas"]:
            saved_filename = metadata.get("saved_filename")

            if saved_filename:
                saved_filenames.add(saved_filename)

        status = get_document_status(file_hash, user_id)
        saved_filename = status.get("saved_filename")

        if saved_filename:
            saved_filenames.add(saved_filename)

        cleanup_document_indexes(user_id, file_hash)
        delete_document_status(file_hash, user_id)
        delete_saved_files(saved_filenames)

        return {"message": "文档已删除"}

    status = get_document_status(file_hash, user_id)

    if not status or status.get("user_id") != user_id or status.get("status") != "failed":
        raise HTTPException(status_code=404, detail="文档不存在")

    saved_filenames = set()
    saved_filename = status.get("saved_filename")

    if saved_filename:
        saved_filenames.add(saved_filename)
    else:
        saved_filenames.update(find_unreferenced_saved_files_by_hash(file_hash))

    cleanup_document_indexes(user_id, file_hash)
    delete_document_status(file_hash, user_id)
    delete_saved_files(saved_filenames)

    return {"message": "文档已删除"}

"""
Redis文档状态管理
"""


def get_document_status_key(file_hash, user_id=None):
    if user_id:
        return f"document:{user_id}:{file_hash}"

    return f"document:{file_hash}"


def set_document_status(file_hash, filename, user_id, status, error=None, saved_filename=None):
    mapping = {
        "file_hash": file_hash,
        "filename": filename,
        "user_id": user_id,
        "status": status,
        "error": error or "",
    }

    if saved_filename:
        mapping["saved_filename"] = saved_filename

    redis_client.hset(get_document_status_key(file_hash, user_id), mapping=mapping)

    legacy_status = get_document_status(file_hash)

    if legacy_status and legacy_status.get("user_id") == user_id:
        redis_client.delete(get_document_status_key(file_hash))

def get_document_status(file_hash, user_id=None):
    if user_id:
        data = redis_client.hgetall(get_document_status_key(file_hash, user_id))

        if data:
            return data

    return redis_client.hgetall(get_document_status_key(file_hash))


def update_document_status(file_hash, status, error=None, user_id=None):
    mapping = {"status": status}

    if error:
        mapping["error"] = error

    status_key = get_document_status_key(file_hash, user_id)

    if user_id and not redis_client.exists(status_key):
        legacy_status = get_document_status(file_hash)

        if legacy_status and legacy_status.get("user_id") == user_id:
            status_key = get_document_status_key(file_hash)

    redis_client.hset(status_key, mapping=mapping)


def delete_document_status(file_hash, user_id=None):
    if user_id:
        redis_client.delete(get_document_status_key(file_hash, user_id))

        legacy_status = get_document_status(file_hash)

        if legacy_status and legacy_status.get("user_id") == user_id:
            redis_client.delete(get_document_status_key(file_hash))

        return

    redis_client.delete(get_document_status_key(file_hash))


def get_all_document_status():

    keys = redis_client.keys("document:*")

    results = []

    for key in keys:

        data = redis_client.hgetall(key)

        if data:
            results.append(data)

    return results


def delete_saved_files(saved_filenames):
    for saved_filename in saved_filenames:
        file_path = os.path.join(DOCS_DIR, os.path.basename(saved_filename))

        if os.path.exists(file_path):
            os.remove(file_path)


def get_referenced_saved_filenames(file_hash):
    docs = vectorstore.get(where={"file_hash": file_hash})
    saved_filenames = set()

    for metadata in docs.get("metadatas", []):
        saved_filename = metadata.get("saved_filename")

        if saved_filename:
            saved_filenames.add(saved_filename)

    return saved_filenames


def calculate_file_md5(file_path):
    file_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            file_md5.update(chunk)

    return file_md5.hexdigest()


def find_unreferenced_saved_files_by_hash(file_hash):
    if not os.path.isdir(DOCS_DIR):
        return set()

    referenced_saved_filenames = get_referenced_saved_filenames(file_hash)
    saved_filenames = set()

    for saved_filename in os.listdir(DOCS_DIR):
        if saved_filename in referenced_saved_filenames:
            continue

        file_path = os.path.join(DOCS_DIR, saved_filename)

        if not os.path.isfile(file_path):
            continue

        try:
            if calculate_file_md5(file_path) == file_hash:
                saved_filenames.add(saved_filename)
        except OSError:
            continue

    return saved_filenames


