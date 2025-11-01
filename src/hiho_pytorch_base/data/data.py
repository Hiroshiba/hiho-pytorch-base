"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor

from hiho_pytorch_base.data.sampling_data import SamplingData


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    feature_vector: numpy.ndarray
    feature_variable: numpy.ndarray
    target_vector: SamplingData
    target_variable: SamplingData
    target_scalar: float
    speaker_id: int


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    feature_vector: Tensor
    feature_variable: Tensor
    target_vector: Tensor
    target_variable: Tensor
    target_scalar: Tensor
    speaker_id: Tensor


def preprocess(
    feature_vector_size: int,
    feature_variable_size: int,
    target_vector_size: int,
    speaker_size: int,
    frame_rate: float,
    frame_length: int,
    is_eval: bool,
) -> OutputData:
    """ランダムダミーデータ生成"""
    variable_length = frame_length

    feature_vector = numpy.full(feature_vector_size, 0.5, dtype=numpy.float32)
    feature_variable = numpy.full((variable_length, feature_variable_size), 0.5, dtype=numpy.float32)

    target_class = 0
    target_variable = numpy.full((variable_length, target_vector_size), 0.5, dtype=numpy.float32)
    target_scalar = 0.5
    speaker_id = 0

    return OutputData(
        feature_vector=torch.from_numpy(feature_vector).float(),
        feature_variable=torch.from_numpy(feature_variable).float(),
        target_vector=torch.tensor(target_class).long(),
        target_variable=torch.from_numpy(target_variable).float(),
        target_scalar=torch.tensor(target_scalar).float(),
        speaker_id=torch.tensor(speaker_id).long(),
    )
