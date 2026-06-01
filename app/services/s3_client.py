from __future__ import annotations

import uuid

import boto3
import requests

from app.core.config import get_settings


def _get_boto_client():
    s = get_settings()
    kwargs: dict = {"region_name": s.aws_region}
    if s.aws_access_key_id:
        kwargs["aws_access_key_id"] = s.aws_access_key_id
        kwargs["aws_secret_access_key"] = s.aws_secret_access_key
    return boto3.client("s3", **kwargs)


def download_pdf(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def upload_pdf(data: bytes, user_id: int | None = None) -> str:
    s = get_settings()
    key = (
        f"files/ai/{user_id}/{uuid.uuid4()}.pdf"
        if user_id
        else f"files/ai/{uuid.uuid4()}.pdf"
    )
    _get_boto_client().put_object(
        Bucket=s.s3_bucket_name,
        Key=key,
        Body=data,
        ContentType="application/pdf",
    )
    return f"https://{s.s3_bucket_name}.s3.{s.aws_region}.amazonaws.com/{key}"
