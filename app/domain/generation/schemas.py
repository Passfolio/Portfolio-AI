from dataclasses import dataclass


@dataclass
class GenerationResult:
    key: str    # 로컬 경로 (dev) / S3 URL (prod 교체 지점)
    size: int   # PDF 바이트 크기
