"""Transformerユーティリティ関数モジュール"""

# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch


def make_pad_mask(length: torch.Tensor):
    """パディングマスクを生成"""
    maxlen = length.max().item()
    mask = torch.arange(maxlen, dtype=torch.int64, device=length.device).unsqueeze(
        0
    ) >= length.unsqueeze(1)
    return mask


def make_non_pad_mask(length: torch.Tensor):
    """非パディングマスクを生成"""
    return ~make_pad_mask(length=length)


def make_length_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """lengthsとmax_lengthから有効部分のマスクを生成（torch.compile対応）"""
    return torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]
