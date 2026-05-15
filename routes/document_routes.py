# 处理文档上传、列出和删除的路由
from fastapi import APIRouter, UploadFile, File, Form

from services.document_service import (
    upload_document,
    list_documents,
    delete_document_by_hash,
)

router = APIRouter()


@router.post("/upload")
async def upload(user_id: str = Form(...), file: UploadFile = File(...)):
    return await upload_document(user_id, file)


@router.get("/documents")
async def documents(user_id: str):
    return list_documents(user_id)


@router.delete("/delete_document")
async def delete_document(user_id: str, file_hash: str):
    return delete_document_by_hash(user_id, file_hash)
