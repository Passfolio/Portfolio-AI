"""공통 PDF 오케스트레이션 유틸리티 — 비동기 RAG 서비스에서 공통 사용."""
from __future__ import annotations

import os
import tempfile
import uuid
from typing import Callable

from app.jobs.store import JobStatus, update_job
from app.services._rag_utils import OUTPUT_DIR
from app.services.s3_client import download_pdf, upload_pdf


def download_pdf_to_temp(s3_url: str) -> str:
    """S3에서 PDF를 다운로드하고 임시 파일 경로를 반환한다."""
    pdf_bytes = download_pdf(s3_url)
    tmp_file  = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path  = tmp_file.name
    tmp_file.write(pdf_bytes)
    tmp_file.close()
    return tmp_path


def make_output_path(suffix: str) -> str:
    """OUTPUT_DIR를 생성하고 고유한 출력 PDF 경로를 반환한다."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return str(OUTPUT_DIR / f"{uuid.uuid4()}_{suffix}.pdf")


def upload_pdf_file(output_path: str, user_id: int | None) -> str:
    """PDF 파일을 읽어 S3에 업로드하고 URL을 반환한다."""
    with open(output_path, "rb") as f:
        return upload_pdf(f.read(), user_id=user_id)


def cleanup_files(*paths: str) -> None:
    """존재하는 파일 경로들을 모두 삭제한다."""
    for path in paths:
        if os.path.exists(path):
            os.unlink(path)


def run_job_pipeline(job_id: str, fn: Callable, tag: str = "") -> None:
    """fn()을 호출하며 RUNNING/DONE/ERROR Job 상태를 관리한다."""
    update_job(job_id, JobStatus.RUNNING)
    try:
        if tag:
            print(f"[{tag}][{job_id}] 시작")
        result = fn()
        if tag:
            sections = result.get("sections", [])
            print(f"[{tag}][{job_id}] 완료: {len(sections)}개 섹션")
        update_job(job_id, JobStatus.DONE, result=result)
    except Exception as e:
        if tag:
            print(f"[{tag}][{job_id}] 오류: {e}")
        update_job(job_id, JobStatus.ERROR, message=str(e))
