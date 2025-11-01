"""バッチ処理モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from hiho_pytorch_base.data.data import OutputData


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    feature_vector: Tensor  # (B, ?)
    feature_variable_nt: Tensor  # NestedTensor (jagged)
    target_vector: Tensor  # (B, ?)
    target_variable_nt: Tensor  # NestedTensor (jagged)
    target_scalar: Tensor  # (B,)
    speaker_id: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.feature_vector.shape[0]

    def to_device(self, device: str, non_blocking: bool) -> Self:
        """データを指定されたデバイスに移動"""
        self.feature_vector = self.feature_vector.to(device, non_blocking=non_blocking)
        self.feature_variable_nt = self.feature_variable_nt.to(
            device, non_blocking=non_blocking
        )
        self.target_vector = self.target_vector.to(device, non_blocking=non_blocking)
        self.target_variable_nt = self.target_variable_nt.to(
            device, non_blocking=non_blocking
        )
        self.target_scalar = self.target_scalar.to(device, non_blocking=non_blocking)
        self.speaker_id = self.speaker_id.to(device, non_blocking=non_blocking)
        return self


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする"""
    return torch.stack(values)


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """DatasetOutputのリストをBatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    feature_variable_nt = torch.nested.nested_tensor(
        [d.feature_variable for d in data_list], layout=torch.jagged
    )
    target_variable_nt = torch.nested.nested_tensor(
        [d.target_variable for d in data_list], layout=torch.jagged
    )

    return BatchOutput(
        feature_vector=collate_stack([d.feature_vector for d in data_list]),
        feature_variable_nt=feature_variable_nt,
        target_vector=collate_stack([d.target_vector for d in data_list]),
        target_variable_nt=target_variable_nt,
        target_scalar=collate_stack([d.target_scalar for d in data_list]),
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
    )
