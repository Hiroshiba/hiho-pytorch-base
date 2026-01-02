"""型ユーティリティ"""

import hashlib
import uuid
from pathlib import Path
from typing import Annotated

from fsspec.implementations.local import LocalFileSystem
from pydantic import BeforeValidator, PlainSerializer
from upath import UPath


def _to_upath(v: str) -> UPath:
    return UPath(v)


def _ser_upath(v: UPath | None) -> str | None:
    return None if v is None else str(v)


UPathField = Annotated[
    UPath,
    BeforeValidator(_to_upath),
    PlainSerializer(_ser_upath, return_type=str),
]


def to_local_path(p: UPath) -> Path:
    """リモートならキャッシュを作ってそのパスを、ローカルならそのままそのパスを返す"""
    if isinstance(p.fs, LocalFileSystem):
        return Path(p)
    cache_dir = Path("hiho_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = str(p)
    cache_key = hashlib.sha256(url.encode()).hexdigest()
    cache_path = cache_dir / cache_key
    if not cache_path.exists():
        tmp_path = cache_dir / f"{cache_key}.{uuid.uuid4().hex}"
        p.fs.get_file(url, str(tmp_path))
        tmp_path.replace(cache_path)
    return cache_path
