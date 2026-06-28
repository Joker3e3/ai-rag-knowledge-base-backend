import logging
import time


def new_timer() -> float:
    return time.perf_counter()


def duration_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def log_rag_timing(
    logger: logging.Logger,
    request_id: str,
    stage: str,
    start: float | None = None,
    duration_ms_value: float | None = None,
    reason: str | None = None,
) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    if duration_ms_value is None:
        if start is None:
            raise ValueError("start is required when duration_ms_value is not provided")
        duration_ms_value = duration_ms(start)

    if reason:
        logger.info(
            "[RAG_TIMING] request_id=%s stage=%s duration_ms=%.1f reason=%s",
            request_id,
            stage,
            duration_ms_value,
            reason,
        )
        return

    logger.info(
        "[RAG_TIMING] request_id=%s stage=%s duration_ms=%.1f",
        request_id,
        stage,
        duration_ms_value,
    )
