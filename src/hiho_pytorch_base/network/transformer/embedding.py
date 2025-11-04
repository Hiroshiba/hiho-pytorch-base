"""Transformer位置エンコーディングモジュール"""

# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """位置エンコーディング"""

    def __init__(
        self, hidden_size: int, dropout_rate: float, cycle_length: float = 10000
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("xscale", torch.tensor(math.sqrt(self.hidden_size)))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.cycle_length = cycle_length
        self.register_buffer("pe", torch.zeros(1, 5000, hidden_size))
        self._extend_pe(5000)

    def _extend_pe(self, target_length: int):
        if self.pe.size(1) >= target_length:
            return
        pe = torch.zeros(
            target_length, self.hidden_size, device=self.pe.device, dtype=self.pe.dtype
        )
        position = torch.arange(
            0, target_length, dtype=torch.float32, device=self.pe.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0, self.hidden_size, 2, dtype=torch.float32, device=self.pe.device
            )
            * -(math.log(self.cycle_length) / self.hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe.to(dtype=self.pe.dtype))

    def forward(self, x: torch.Tensor):  # noqa: D102
        self._extend_pe(x.size(1))
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RelPositionalEncoding(nn.Module):
    """相対位置エンコーディング"""

    def __init__(
        self, hidden_size: int, dropout_rate: float, cycle_length: float = 10000
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("xscale", torch.tensor(math.sqrt(self.hidden_size)))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.cycle_length = cycle_length
        self.register_buffer("pe", torch.zeros(1, 5000 * 2 - 1, hidden_size))
        self._extend_pe(5000)

    def _extend_pe(self, target_length: int):
        if self.pe.size(1) >= target_length * 2 - 1:
            return
        pe_positive = torch.zeros(
            target_length, self.hidden_size, device=self.pe.device, dtype=self.pe.dtype
        )
        pe_negative = torch.zeros(
            target_length, self.hidden_size, device=self.pe.device, dtype=self.pe.dtype
        )
        position = torch.arange(
            0, target_length, dtype=torch.float32, device=self.pe.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0, self.hidden_size, 2, dtype=torch.float32, device=self.pe.device
            )
            * -(math.log(self.cycle_length) / self.hidden_size)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.register_buffer("pe", pe.to(dtype=self.pe.dtype))

    def forward(  # noqa: D102
        self,
        x: Tensor,  # (B, T, ?)
    ):
        self._extend_pe(x.size(1))
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)
