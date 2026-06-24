"""DataLoader作成モジュール"""

import math
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future

from torch.utils.data import BatchSampler, DataLoader, Sampler

from .batch import collate_dataset_output
from .dataset import Dataset


class PreparingBatchSampler(Sampler[list[int]]):
    """実際のサンプリング順に先行してファイルを準備するBatchSampler"""

    def __init__(
        self,
        dataset: Dataset,
        batch_sampler: BatchSampler,
        buffered_batch_num: int,
    ) -> None:
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.buffered_batch_num = buffered_batch_num

    def __len__(self) -> int:
        """バッチ数を返す"""
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[list[int]]:
        """サンプルを先行ダウンロードしながらバッチを生成する"""
        source = iter(self.batch_sampler)
        pending: deque[tuple[list[int], Future[None]]] = deque()

        def try_add_next() -> bool:
            try:
                batch = list(next(source))
            except StopIteration:
                return False
            pending.append((batch, self.dataset.prepare(batch)))
            return True

        for _ in range(self.buffered_batch_num):
            if not try_add_next():
                break

        while pending:
            batch, prepared = pending.popleft()
            prepared.result()
            try_add_next()
            yield batch


def create_prepared_data_loader(
    dataset: Dataset,
    batch_size: int,
    sampler: Sampler[int],
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    """ファイル先行ダウンロード付きのDataLoaderを作成する"""
    batch_sampler = BatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    download_buffer = math.ceil(
        dataset.file_cache.max_concurrent_downloads / max(batch_size, 1)
    )
    data_loader_buffer = num_workers * 2 + 1
    buffered_batch_num = max(1, download_buffer, data_loader_buffer)

    preparing_sampler = PreparingBatchSampler(
        dataset=dataset,
        batch_sampler=batch_sampler,
        buffered_batch_num=buffered_batch_num,
    )

    return DataLoader(
        dataset=dataset,
        batch_sampler=preparing_sampler,
        num_workers=num_workers,
        collate_fn=collate_dataset_output,
        pin_memory=pin_memory,
        timeout=0 if num_workers == 0 else 300,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
