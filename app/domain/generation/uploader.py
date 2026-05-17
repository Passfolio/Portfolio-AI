from pathlib import Path


def upload(pdf_bytes: bytes, key: str) -> str:
    path = Path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_bytes)
    return str(path)
