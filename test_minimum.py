"""最小構成でセグフォ/ハングを再現 - EncoderLayerクラスを使用"""

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from hiho_pytorch_base.network.conformer.convolution import ConvGLUModule
from hiho_pytorch_base.network.conformer.encoder import Swish
from hiho_pytorch_base.network.conformer.encoder_layer import EncoderLayer
from hiho_pytorch_base.network.transformer.attention import (
    RelPositionMultiHeadedAttention,
)
from hiho_pytorch_base.network.transformer.multi_layer_conv import FastSpeechTwoConv


@torch.compile(dynamic=True, mode="reduce-overhead")
def train_step(encoder_layers, scaler, x, pos_emb, mask):
    with autocast("cuda", enabled=True):
        for encoder_layer in encoder_layers:
            x, pos_emb, mask = encoder_layer(x=x, pos_emb=pos_emb, mask=mask)
        loss = x.sum()

    scaler.scale(loss).backward()
    return loss


def main():
    print("Starting test with EncoderLayer class", flush=True)
    device = "cuda"
    batch_size = 4
    seq_len = 50
    hidden_size = 32
    num_layers = 2

    print(f"Creating {num_layers} EncoderLayers...", flush=True)
    encoder_layers = []

    for i in range(num_layers):
        encoder_layer = EncoderLayer(
            hidden_size=hidden_size,
            self_attn=RelPositionMultiHeadedAttention(
                head_size=8,
                hidden_size=hidden_size,
                dropout_rate=0.1,
            ),
            conv_module=ConvGLUModule(
                hidden_size=hidden_size,
                kernel_size=31,
                activation=Swish(),
            ),
            macaron_feed_forward=FastSpeechTwoConv(
                inout_size=hidden_size,
                hidden_size=hidden_size * 4,
                kernel_size=3,
                dropout_rate=0.1,
            ),
            feed_forward=FastSpeechTwoConv(
                inout_size=hidden_size,
                hidden_size=hidden_size * 4,
                kernel_size=3,
                dropout_rate=0.1,
            ),
            dropout_rate=0.1,
        ).to(device)
        encoder_layers.append(encoder_layer)

    print("EncoderLayers created", flush=True)

    all_params = []
    for encoder_layer in encoder_layers:
        all_params.extend(list(encoder_layer.parameters()))

    optimizer = torch.optim.SGD(all_params, lr=0.001)
    scaler = GradScaler(device, enabled=True)
    print("Optimizer created", flush=True)

    print("Starting training loop...", flush=True)
    for step in range(10):
        print(f"Step {step + 1} starting...", flush=True)
        x = torch.randn(
            batch_size, seq_len, hidden_size, device=device, requires_grad=True
        )
        pos_emb = torch.randn(1, 2 * seq_len - 1, hidden_size, device=device)
        mask = torch.ones(batch_size, 1, seq_len, device=device)

        loss = train_step(encoder_layers, scaler, x, pos_emb, mask)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        print(f"Step {step + 1}/10 completed, loss: {loss.item():.4f}", flush=True)

    print("Test completed successfully!")


if __name__ == "__main__":
    main()
