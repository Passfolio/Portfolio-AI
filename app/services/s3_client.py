from __future__ import annotations

import uuid
from urllib.parse import urlparse

import boto3

from app.core.config import get_settings


def _get_boto_client():
    s = get_settings()
    kwargs: dict = {"region_name": s.aws_region}
    if s.aws_access_key_id:
        kwargs["aws_access_key_id"] = s.aws_access_key_id
        kwargs["aws_secret_access_key"] = s.aws_secret_access_key
    return boto3.client("s3", **kwargs)


def _parse_s3_url(url: str) -> tuple[str, str]:
    """S3 URL을 (bucket, key) 튜플로 파싱."""
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    host = parsed.hostname or ""
    if host.startswith("s3"):
        # path-style: https://s3.region.amazonaws.com/bucket/key
        parts = path.split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""
    # virtual-hosted: https://bucket.s3.region.amazonaws.com/key
    return host.split(".")[0], path


def download_pdf(s3_url: str) -> bytes:
    bucket, key = _parse_s3_url(s3_url)
    resp = _get_boto_client().get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def upload_pdf(data: bytes, user_id: int | None = None) -> str:
    s = get_settings()
    key = (
        f"outputs/pdfs/{user_id}/{uuid.uuid4()}.pdf"
        if user_id
        else f"outputs/pdfs/{uuid.uuid4()}.pdf"
    )
    _get_boto_client().put_object(
        Bucket=s.s3_bucket_name,
        Key=key,
        Body=data,
        ContentType="application/pdf",
    )
    return f"https://{s.s3_bucket_name}.s3.{s.aws_region}.amazonaws.com/{key}"
