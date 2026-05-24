"""공통 PDF 오케스트레이션 유틸리티 — 비동기 RAG 서비스에서 공통 사용."""
from __future__ import annotations

import logging
import os
import tempfile
import uuid
from typing import Callable

from app.jobs.store import JobStatus, update_job
from app.services._rag_utils import OUTPUT_DIR
from app.services.s3_client import download_pdf, upload_pdf

logger = logging.getLogger(__name__)


def download_pdf_to_temp(s3_url: str) -> str:
    """S3에서 PDF를 다운로드하고 임시 파일 경로를 반환한다."""
    logger.info("PDF 다운로드 시작: %s", s3_url)
    pdf_bytes = download_pdf(s3_url)
    tmp_file  = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path  = tmp_file.name
    tmp_file.write(pdf_bytes)
    tmp_file.close()
    logger.info("PDF 다운로드 완료: %s (%d bytes)", tmp_path, len(pdf_bytes))
    return tmp_path


def make_output_path(suffix: str) -> str:
    """OUTPUT_DIR를 생성하고 고유한 출력 PDF 경로를 반환한다."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return str(OUTPUT_DIR / f"{uuid.uuid4()}_{suffix}.pdf")


def upload_pdf_file(output_path: str, user_id: int | None) -> str:
    """PDF 파일을 읽어 S3에 업로드하고 URL을 반환한다."""
    logger.info("S3 업로드 시작: %s", output_path)
    with open(output_path, "rb") as f:
        url = upload_pdf(f.read(), user_id=user_id)
    logger.info("S3 업로드 완료: %s", url)
    return url


def cleanup_files(*paths: str) -> None:
    """존재하는 파일 경로들을 모두 삭제한다."""
    for path in paths:
        if os.path.exists(path):
            os.unlink(path)


def run_job_pipeline(job_id: str, fn: Callable, tag: str = "") -> None:
    """fn()을 호출하며 RUNNING/DONE/ERROR Job 상태를 관리한다."""
    logger.info("[%s][%s] Job 시작 (RUNNING)", tag, job_id)
    update_job(job_id, JobStatus.RUNNING)
    try:
        result = fn()
        sections = result.get("sections", [])
        logger.info("[%s][%s] Job 완료 (DONE): %d개 섹션", tag, job_id, len(sections))
        update_job(job_id, JobStatus.DONE, result=result)
    except Exception as e:
        logger.error("[%s][%s] Job 오류 (ERROR): %s", tag, job_id, e)
        update_job(job_id, JobStatus.ERROR, message=str(e))
