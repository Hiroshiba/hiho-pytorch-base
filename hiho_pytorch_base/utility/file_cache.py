"""リモートファイルのキャッシュ管理モジュール"""

import hashlib
import os
import tempfile
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from filelock import FileLock
from fsspec.implementations.local import LocalFileSystem
from upath import UPath


def _completed_future[T](value: T) -> "Future[T]":
    """完了済みのFutureを返す"""
    f: Future[T] = Future()
    f.set_result(value)
    return f


def _gather_to_none(futures: "list[Future[Any]]") -> "Future[None]":
    """複数のFutureをまとめ、全て完了したときにNoneで完了するFutureを返す"""
    if not futures:
        return _completed_future(None)

    result: Future[None] = Future()
    remaining = [len(futures)]
    lock = threading.Lock()

    def on_done(f: "Future[Any]") -> None:
        if result.done():
            return
        exc = f.exception()
        if exc is not None:
            try:
                result.set_exception(exc)
            except Exception:
                pass
            return
        with lock:
            remaining[0] -= 1
            if remaining[0] == 0:
                try:
                    result.set_result(None)
                except Exception:
                    pass

    for f in futures:
        f.add_done_callback(on_done)

    return result


class RemoteFileCache:
    """リモートファイルをローカルへ原子的にキャッシュする"""

    def __init__(self, cache_dir: Path, max_concurrent_downloads: int) -> None:
        self._cache_dir = cache_dir
        self.max_concurrent_downloads = max_concurrent_downloads
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_downloads,
            thread_name_prefix="remote-file-cache",
        )
        self._runtime_lock = threading.Lock()
        self._inflight: dict[str, Future[Path | None]] = {}

    def __getstate__(self) -> dict[str, Any]:
        """pickleシリアライズ用の状態を返す"""
        return {
            "cache_dir": self._cache_dir,
            "max_concurrent_downloads": self.max_concurrent_downloads,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """pickle復元時にスレッドリソースを再初期化する"""
        self._cache_dir = state["cache_dir"]
        self.max_concurrent_downloads = state["max_concurrent_downloads"]
        self._executor = ThreadPoolExecutor(
            max_workers=state["max_concurrent_downloads"],
            thread_name_prefix="remote-file-cache",
        )
        self._runtime_lock = threading.Lock()
        self._inflight = {}

    def is_local(self, path: UPath) -> bool:
        """ローカルファイルシステムかどうかを判定する"""
        return isinstance(path.fs, LocalFileSystem)

    def _cache_path_for(self, path: UPath) -> Path:
        """リモートパスに対応するローカルキャッシュパスを返す"""
        digest = hashlib.sha256(str(path).encode()).hexdigest()
        return self._cache_dir / "files" / f"{digest}{path.suffix}"

    def _lock_path_for(self, key: str) -> Path:
        """ファイルロックのパスを返す"""
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / "locks" / f"{digest}.lock"

    def _create_temp_file(self, parent: Path) -> Path:
        """一時ファイルを作成する"""
        with tempfile.NamedTemporaryFile(dir=parent, delete=False) as tmp:
            return Path(tmp.name)

    def _download(self, path: UPath) -> Path | None:
        """リモートファイルをダウンロードしてローカルキャッシュパスを返す。存在しなければNoneを返す"""
        destination = self._cache_path_for(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        lock_path = self._lock_path_for(str(path))
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with FileLock(str(lock_path)):
            if destination.is_file():
                return destination

            tmp = self._create_temp_file(destination.parent)
            try:
                path.fs.get_file(path.path, str(tmp))
                os.replace(tmp, destination)
            except FileNotFoundError:
                return None
            finally:
                tmp.unlink(missing_ok=True)

        return destination

    def submit_optional(self, path: UPath) -> "Future[Path | None]":
        """リモートファイルのダウンロードをサブミットする。存在しなければNoneで完了する"""
        if self.is_local(path):
            return _completed_future(Path(path))

        destination = self._cache_path_for(path)
        if destination.is_file():
            return _completed_future(destination)

        key = str(path)

        with self._runtime_lock:
            existing = self._inflight.get(key)
            if existing is not None:
                return existing

            future: Future[Path | None] = self._executor.submit(self._download, path)
            self._inflight[key] = future

        def remove_inflight(_: "Future[Path | None]") -> None:
            with self._runtime_lock:
                if self._inflight.get(key) is future:
                    del self._inflight[key]

        future.add_done_callback(remove_inflight)
        return future

    def submit(self, path: UPath) -> "Future[Path]":
        """リモートファイルのダウンロードをサブミットする。存在しなければ例外を投げる"""
        optional_future = self.submit_optional(path)
        result: Future[Path] = Future()

        def on_done(f: "Future[Path | None]") -> None:
            if result.done():
                return
            exc = f.exception()
            if exc is not None:
                try:
                    result.set_exception(exc)
                except Exception:
                    pass
                return
            local_path = f.result()
            if local_path is None:
                try:
                    result.set_exception(
                        FileNotFoundError(f"ファイルが存在しません: {path}")
                    )
                except Exception:
                    pass
                return
            try:
                result.set_result(local_path)
            except Exception:
                pass

        optional_future.add_done_callback(on_done)
        return result

    def download(self, path: UPath) -> Path:
        """リモートファイルをダウンロードしてローカルパスを返す"""
        return self.submit(path).result()

    def download_many(self, paths: Sequence[UPath]) -> list[Path]:
        """複数のリモートファイルをダウンロードしてローカルパスを返す"""
        futures = [self.submit(p) for p in paths]
        return [f.result() for f in futures]

    def is_materialized(self, path: UPath) -> bool:
        """ローカルまたはキャッシュ済みかどうかを判定する"""
        if self.is_local(path):
            return Path(path).is_file()
        return self._cache_path_for(path).is_file()

    def require_local_path(self, path: UPath) -> Path:
        """ローカルパスを返す。キャッシュが存在しなければ例外を投げる"""
        if self.is_local(path):
            return Path(path)
        destination = self._cache_path_for(path)
        if not destination.is_file():
            raise FileNotFoundError(f"キャッシュが存在しません: {path}")
        return destination

    def publish_generated(self, path: UPath, generate: Callable[[Path], None]) -> Path:
        """生成処理を実行して結果を公開する"""
        if self.is_local(path):
            return self._publish_local(Path(path), generate)
        return self._publish_remote(path, generate)

    def _publish_local(
        self, destination: Path, generate: Callable[[Path], None]
    ) -> Path:
        """ローカルファイルを原子的に生成して公開する"""
        destination.parent.mkdir(parents=True, exist_ok=True)

        lock_path = self._lock_path_for(f"publish_{destination}")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with FileLock(str(lock_path)):
            if destination.is_file():
                return destination

            tmp = self._create_temp_file(destination.parent)
            try:
                generate(tmp)
                os.replace(tmp, destination)
            finally:
                tmp.unlink(missing_ok=True)

        return destination

    def _publish_remote(self, path: UPath, generate: Callable[[Path], None]) -> Path:
        """リモートファイルを生成して公開し、ローカルキャッシュにも保存する"""
        destination = self._cache_path_for(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        lock_path = self._lock_path_for(f"publish_{path}")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with FileLock(str(lock_path)):
            if destination.is_file():
                return destination

            tmp = self._create_temp_file(destination.parent)
            try:
                generate(tmp)
                path.fs.put_file(str(tmp), path.path)
                os.replace(tmp, destination)
            finally:
                tmp.unlink(missing_ok=True)

        return destination

    def close(self) -> None:
        """ダウンロードサービスを停止する"""
        self._executor.shutdown(wait=True, cancel_futures=True)
