"""最小再現コード: Python float属性 + 複数モジュール + torch.compile でハング"""

import torch
from torch import nn


class MinimalModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        self.linear = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.1)
        self.scale = 0.5

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = residual + self.scale * self.dropout(x)
        return x


@torch.compile(dynamic=True, mode="reduce-overhead")
def train_step(layers, x):
    for layer in layers:
        x = layer(x)
    loss = x.sum()
    loss.backward()
    return loss


def main():
    print("Testing with 2 layers, 1 norm, 1 linear...", flush=True)
    layers = nn.ModuleList([MinimalModule().cuda() for _ in range(2)])
    x = torch.randn(4, 50, 32, device="cuda", requires_grad=True)

    print("Running train_step...", flush=True)
    loss = train_step(layers, x)
    print(f"Completed! Loss: {loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()
