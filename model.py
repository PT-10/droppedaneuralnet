import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return x + self.out(self.activation(self.inp(x)))


class Model(nn.Module):
    def __init__(self, num_blocks=48):
        super().__init__()
        self.blocks = nn.ModuleList([Block(48, 96) for _ in range(num_blocks)])
        self.final = nn.Linear(48, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final(x)
