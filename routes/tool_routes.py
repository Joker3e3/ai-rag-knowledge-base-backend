import logging
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config.rag_config import (
    BM25_WEIGHT,
    COMPRESS_CONTEXT,
    RECALL_K,
    RERANK_TOP_K,
    VECTOR_WEIGHT,
)
from database.database import SessionLocal, get_db
from database.models.candidate import Candidate
from database.models.resume import Resume
from retrievers.custom_retriever import CustomRerankRetriever
from schemas.tool_schema import ToolQuery
from services.rag_service import llm, vectorstore
from utils.rag_timing import log_rag_timing, new_timer


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/retrieve_evidence")
def retrieve_evidence(query: ToolQuery):
    request_id = uuid.uuid4().hex[:8]
    total_start = new_timer()

    retriever = CustomRerankRetriever(
        vectorstore=vectorstore,
        user_id=query.user_id,
        candidate_id=query.candidate_id,
        resume_id=query.resume_id,
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

@router.get("/candidates")
def list_candidates(user_id: str, db: Session = Depends(get_db)):
    db = SessionLocal()
    candidates = (
        db.query(Candidate)
        .filter(Candidate.user_id == user_id)
        .all()
    )

    return [
        {
            "candidate_id": c.candidate_id,
            "name": c.name,
            "school": c.school,
            "phone": c.phone,
        }
        for c in candidates
    ]

@router.get("/candidates/{candidate_id}/resumes")
def list_resumes(candidate_id: str, db: Session = Depends(get_db)):
    resumes = (
        db.query(Resume)
        .filter(Resume.candidate_id == candidate_id)
        .order_by(Resume.created_at.desc())
        .all()
    )

    return [
        {
            "resume_id": r.resume_id,
            "candidate_id": r.candidate_id,
            "file_name": r.file_name,
            "file_hash": r.file_hash,
            "is_latest": r.is_latest,
            "created_at": r.created_at,
        }
        for r in resumes
    ]
