import os

from dotenv import load_dotenv


load_dotenv()


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off; got {value!r}"
    )


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RECALL_K = 10
RERANK_TOP_K = 3
# Benchmark switch for CPU/scoped retrieval A/B tests. Keep enabled by default
# so production behavior remains unchanged unless RERANK_ENABLED=false is set.
RERANK_ENABLED = _parse_bool_env("RERANK_ENABLED", True)

BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5

COMPRESS_CONTEXT = False
REWRITE_QUERY = None

DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
