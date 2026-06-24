"""HDF5キャッシュモジュール"""

from pathlib import Path

import h5py
import numpy

from .data import InputData
from .sampling_data import SamplingData

HDF5_CACHE_VERSION = 1


def write_hdf5_cache(path: Path, data: InputData, manifest: dict[str, str]) -> None:
    """InputDataをHDF5キャッシュファイルに書き込む"""
    with h5py.File(path, "w") as f:
        f.attrs["cache_version"] = HDF5_CACHE_VERSION
        for key, value in manifest.items():
            f.attrs[f"manifest_{key}"] = value
        f.create_dataset("feature_vector", data=data.feature_vector)
        f.create_dataset("feature_variable", data=data.feature_variable)
        f.create_dataset("target_vector_array", data=data.target_vector.array)
        f.create_dataset("target_vector_rate", data=data.target_vector.rate)
        f.create_dataset("target_variable_array", data=data.target_variable.array)
        f.create_dataset("target_variable_rate", data=data.target_variable.rate)
        f.create_dataset("target_scalar", data=data.target_scalar)


def read_hdf5_cache(
    path: Path, speaker_id: int, expected_manifest: dict[str, str]
) -> InputData:
    """HDF5キャッシュファイルからInputDataを読み込む"""
    with h5py.File(path, "r") as f:
        version = int(f.attrs["cache_version"])  # type: ignore[arg-type]
        if version != HDF5_CACHE_VERSION:
            raise ValueError(
                f"HDF5キャッシュのバージョンが一致しません: expected={HDF5_CACHE_VERSION} actual={version}"
            )
        for key, expected in expected_manifest.items():
            actual = str(f.attrs[f"manifest_{key}"])
            if actual != expected:
                raise ValueError(f"HDF5キャッシュのmanifestが一致しません: key={key}")
        return InputData(
            feature_vector=numpy.array(f["feature_vector"]),
            feature_variable=numpy.array(f["feature_variable"]),
            target_vector=SamplingData(
                array=numpy.array(f["target_vector_array"]),
                rate=float(numpy.array(f["target_vector_rate"])),
            ),
            target_variable=SamplingData(
                array=numpy.array(f["target_variable_array"]),
                rate=float(numpy.array(f["target_variable_rate"])),
            ),
            target_scalar=float(numpy.array(f["target_scalar"])),
            speaker_id=speaker_id,
        )
