"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, mse_loss

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.config import ModelConfig
from hiho_pytorch_base.network.predictor import Predictor
from hiho_pytorch_base.network.transformer.utility import make_length_mask
from hiho_pytorch_base.utility.pytorch_utility import detach_cpu
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    loss_vector: Tensor
    loss_variable: Tensor
    loss_scalar: Tensor
    accuracy: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.loss_vector = detach_cpu(self.loss_vector)
        self.loss_variable = detach_cpu(self.loss_variable)
        self.loss_scalar = detach_cpu(self.loss_scalar)
        self.accuracy = detach_cpu(self.accuracy)
        return self


def accuracy(
    output: Tensor,  # (B, ?)
    target: Tensor,  # (B,)
) -> Tensor:
    """分類精度を計算"""
    with torch.no_grad():
        indexes = torch.argmax(output, dim=1)  # (B,)
        correct = torch.eq(indexes, target).view(-1)  # (B,)
        return correct.float().mean()


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        (
            vector_output,  # (B, ?)
            variable_output_padded,  # (B, L, ?)
            scalar_output,  # (B,)
        ) = self.predictor(
            feature_vector=batch.feature_vector,
            feature_variable_nt=batch.feature_variable_nt,
            speaker_id=batch.speaker_id,
        )

        target_vector = batch.target_vector  # (B,)
        target_scalar = batch.target_scalar  # (B,)

        target_variable_padded = batch.target_variable_nt.to_padded_tensor(0.0)

        max_length = variable_output_padded.size(1)
        target_variable_lengths = batch.target_variable_nt.offsets().diff()
        mask = make_length_mask(target_variable_lengths, max_length)  # (B, L)
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)

        masked_squared_diff = (variable_output_padded - target_variable_padded).pow(2) * mask_expanded
        loss_variable = masked_squared_diff.sum() / mask.sum()

        loss_vector = cross_entropy(vector_output, target_vector)
        loss_scalar = mse_loss(scalar_output, target_scalar)
        total_loss = loss_vector + loss_variable + loss_scalar
        acc = accuracy(vector_output, target_vector)

        return ModelOutput(
            loss=total_loss,
            loss_vector=loss_vector,
            loss_variable=loss_variable,
            loss_scalar=loss_scalar,
            accuracy=acc,
            data_num=batch.data_num,
        )
