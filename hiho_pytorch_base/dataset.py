"""データセットモジュール"""

import hashlib
import random
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import assert_never

import numpy
from pydantic import TypeAdapter
from torch.utils.data import Dataset as BaseDataset
from upath import UPath

from .config import DataFileConfig, DatasetConfig
from .data.data import InputData, OutputData, preprocess
from .data.hdf5_cache import HDF5_CACHE_VERSION, read_hdf5_cache, write_hdf5_cache
from .data.sampling_data import SamplingData
from .utility.file_cache import RemoteFileCache, _completed_future, _gather_to_none
from .utility.pathlist_utility import get_data_paths


@dataclass
class LazyInputData:
    """遅延読み込み対応の入力データ構造"""

    feature_vector_path: UPath
    feature_variable_path: UPath
    target_vector_path: UPath
    target_variable_path: UPath
    target_scalar_path: UPath
    speaker_id: int
    hdf5_cache_dir: UPath | None

    def source_paths(self) -> tuple[UPath, ...]:
        """ソースファイルのパスを返す"""
        return (
            self.feature_vector_path,
            self.feature_variable_path,
            self.target_vector_path,
            self.target_variable_path,
            self.target_scalar_path,
        )

    def _generate_hdf5_cache_filename(self) -> str:
        key = "\n".join(
            [
                f"version={HDF5_CACHE_VERSION}",
                *(str(path) for path in self.source_paths()),
            ]
        )
        return f"{hashlib.sha256(key.encode()).hexdigest()}.h5"

    def _get_hdf5_cache_path(self) -> UPath:
        if self.hdf5_cache_dir is None:
            raise RuntimeError("hdf5_cache_dirがNoneです")
        return self.hdf5_cache_dir / self._generate_hdf5_cache_filename()

    def _get_manifest(self) -> dict[str, str]:
        return {
            "feature_vector_path": str(self.feature_vector_path),
            "feature_variable_path": str(self.feature_variable_path),
            "target_vector_path": str(self.target_vector_path),
            "target_variable_path": str(self.target_variable_path),
            "target_scalar_path": str(self.target_scalar_path),
        }

    def prepare(self, file_cache: RemoteFileCache) -> "Future[None]":
        """ファイルのダウンロードをサブミットし、完了時にNoneで解決するFutureを返す"""
        if self.hdf5_cache_dir is None:
            return _gather_to_none(
                [file_cache.submit_optional(p) for p in self.source_paths()]
            )

        hdf5_path = self._get_hdf5_cache_path()

        if file_cache.is_local(hdf5_path):
            if Path(hdf5_path).is_file():
                return _completed_future(None)
            return _gather_to_none(
                [file_cache.submit_optional(p) for p in self.source_paths()]
            )

        if file_cache.is_materialized(hdf5_path):
            return _completed_future(None)

        hdf5_future = file_cache.submit_optional(hdf5_path)
        result: Future[None] = Future()

        def on_hdf5_done(f: "Future[Path | None]") -> None:
            if result.done():
                return
            exc = f.exception()
            if exc is not None:
                try:
                    result.set_exception(exc)
                except Exception:
                    pass
                return

            if f.result() is not None:
                try:
                    result.set_result(None)
                except Exception:
                    pass
                return

            raw_gather = _gather_to_none(
                [file_cache.submit_optional(p) for p in self.source_paths()]
            )

            def on_raw_done(raw_f: "Future[None]") -> None:
                if result.done():
                    return
                raw_exc = raw_f.exception()
                if raw_exc is not None:
                    try:
                        result.set_exception(raw_exc)
                    except Exception:
                        pass
                    return
                try:
                    result.set_result(None)
                except Exception:
                    pass

            raw_gather.add_done_callback(on_raw_done)

        hdf5_future.add_done_callback(on_hdf5_done)
        return result

    def _fetch_from_files(self, file_cache: RemoteFileCache) -> InputData:
        paths = file_cache.download_many(list(self.source_paths()))
        (
            feature_vector_path,
            feature_variable_path,
            target_vector_path,
            target_variable_path,
            target_scalar_path,
        ) = paths
        return InputData(
            feature_vector=numpy.load(feature_vector_path, allow_pickle=True),
            feature_variable=numpy.load(feature_variable_path, allow_pickle=True),
            target_vector=SamplingData.load(target_vector_path),
            target_variable=SamplingData.load(target_variable_path),
            target_scalar=float(numpy.load(target_scalar_path, allow_pickle=True)),
            speaker_id=self.speaker_id,
        )

    def fetch(self, file_cache: RemoteFileCache) -> InputData:
        """ファイルからデータを読み込んでInputDataを生成"""
        if self.hdf5_cache_dir is None:
            return self._fetch_from_files(file_cache)

        hdf5_path = self._get_hdf5_cache_path()

        if file_cache.is_materialized(hdf5_path):
            return read_hdf5_cache(
                file_cache.require_local_path(hdf5_path),
                self.speaker_id,
                self._get_manifest(),
            )

        input_data = self._fetch_from_files(file_cache)
        manifest = self._get_manifest()

        def generate_hdf5(tmp_path: Path) -> None:
            write_hdf5_cache(tmp_path, input_data, manifest)

        file_cache.publish_generated(hdf5_path, generate_hdf5)
        return input_data


class Dataset(BaseDataset[OutputData]):
    """メインのデータセット"""

    def __init__(
        self,
        datas: list[LazyInputData],
        config: DatasetConfig,
        is_eval: bool,
        file_cache: RemoteFileCache,
    ):
        self.datas = datas
        self.config = config
        self.is_eval = is_eval
        self.file_cache = file_cache

    def prepare(self, indices: Sequence[int]) -> "Future[None]":
        """指定インデックスのデータのダウンロードをサブミットする"""
        futures = [self.datas[i].prepare(self.file_cache) for i in indices]
        return _gather_to_none(futures)

    def __len__(self) -> int:
        """データセットのサイズ"""
        return len(self.datas)

    def __getitem__(self, i: int) -> OutputData:
        """指定されたインデックスのデータを前処理して返す"""
        try:
            return preprocess(
                self.datas[i].fetch(self.file_cache),
                frame_rate=self.config.frame_rate,
                frame_length=self.config.frame_length,
                is_eval=self.is_eval,
            )
        except Exception as e:
            raise RuntimeError(
                f"データ処理に失敗しました: index={i} data={self.datas[i]}"
            ) from e


class DatasetType(str, Enum):
    """データセットタイプ"""

    TRAIN = "train"
    TEST = "test"
    EVAL = "eval"
    VALID = "valid"


@dataclass
class DatasetCollection:
    """データセットコレクション"""

    train: Dataset
    """重みの更新に用いる"""

    test: Dataset
    """trainと同じドメインでモデルの過適合確認に用いる"""

    eval: Dataset | None
    """testと同じデータを評価に用いる"""

    valid: Dataset | None
    """trainやtestと異なり、評価専用に用いる"""

    file_cache: RemoteFileCache
    """リモートファイルキャッシュ"""

    def close(self) -> None:
        """リソースを解放する"""
        self.file_cache.close()

    def get(self, type: DatasetType) -> Dataset:
        """指定されたタイプのデータセットを返す"""
        match type:
            case DatasetType.TRAIN:
                return self.train
            case DatasetType.TEST:
                return self.test
            case DatasetType.EVAL:
                if self.eval is None:
                    raise ValueError("evalデータセットが設定されていません")
                return self.eval
            case DatasetType.VALID:
                if self.valid is None:
                    raise ValueError("validデータセットが設定されていません")
                return self.valid
            case _:
                assert_never(type)


def get_datas(
    config: DataFileConfig,
    hdf5_cache_dir: UPath | None,
    file_cache: RemoteFileCache,
) -> list[LazyInputData]:
    """データを取得"""
    (
        fn_list,
        (
            feature_vector_pathmappings,
            feature_variable_pathmappings,
            target_vector_pathmappings,
            target_variable_pathmappings,
            target_scalar_pathmappings,
        ),
    ) = get_data_paths(
        config.root_dir,
        [
            config.feature_vector_pathlist_path,
            config.feature_variable_pathlist_path,
            config.target_vector_pathlist_path,
            config.target_variable_pathlist_path,
            config.target_scalar_pathlist_path,
        ],
        file_cache=file_cache,
    )

    fn_each_speaker = TypeAdapter(dict[str, list[str]]).validate_json(
        file_cache.download(config.speaker_dict_path).read_text()
    )
    speaker_ids = {
        fn: speaker_id
        for speaker_id, fns in enumerate(fn_each_speaker.values())
        for fn in fns
    }

    datas = [
        LazyInputData(
            feature_vector_path=feature_vector_pathmappings[fn],
            feature_variable_path=feature_variable_pathmappings[fn],
            target_vector_path=target_vector_pathmappings[fn],
            target_variable_path=target_variable_pathmappings[fn],
            target_scalar_path=target_scalar_pathmappings[fn],
            speaker_id=speaker_ids[fn],
            hdf5_cache_dir=hdf5_cache_dir,
        )
        for fn in fn_list
    ]
    return datas


def create_dataset(config: DatasetConfig) -> DatasetCollection:
    """データセットを作成"""
    file_cache = RemoteFileCache(
        cache_dir=config.local_cache_dir,
        max_concurrent_downloads=config.max_concurrent_downloads,
    )
    try:
        datas = get_datas(config.train, config.hdf5_cache_dir, file_cache)

        if config.seed is not None:
            random.Random(config.seed).shuffle(datas)

        tests, trains = datas[: config.test_num], datas[config.test_num :]
        trains = trains[: config.train_num]

        def _wrap(datas: list[LazyInputData], is_eval: bool) -> Dataset:
            if is_eval:
                datas = datas * config.eval_times_num
            return Dataset(
                datas=datas, config=config, is_eval=is_eval, file_cache=file_cache
            )

        return DatasetCollection(
            train=_wrap(trains, is_eval=False),
            test=_wrap(tests, is_eval=False),
            eval=(_wrap(tests, is_eval=True) if config.eval_for_test else None),
            valid=(
                _wrap(
                    get_datas(config.valid, config.hdf5_cache_dir, file_cache)[
                        : config.valid_num
                    ],
                    is_eval=True,
                )
                if config.valid is not None
                else None
            ),
            file_cache=file_cache,
        )
    except BaseException:
        file_cache.close()
        raise
