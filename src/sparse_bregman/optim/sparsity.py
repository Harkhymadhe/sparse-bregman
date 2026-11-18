# -*- coding: utf-8 -*-

import torch
from torch import nn

from sparse_bregman.optim.regularizers import *

__all__ = [
    "L1L2SparsityEnforcer",
    "L1SparsityEnforcer",
    "NullSparsityEnforcer",
    "SparsityEnforcer",
]


class SparsityEnforcer(nn.Module):
    def __init__(self, weight: float = 1.0, requires_grad: bool = False):
        super().__init__()

        self.weight = nn.Parameter(
            torch.tensor(weight, dtype=torch.float32), requires_grad=requires_grad
        )

    def proximity_operator(self, x: torch.Tensor, delta: float = 1.0):
        raise NotImplementedError("Subclasses must implement the proximity operator.")

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def sub_gradient(self, v: torch.Tensor):
        return torch.zeros_like(v)

    def initialize_sub_gradient(self, x: torch.Tensor, delta: float = 1.0):
        x_d = x.data.clone()
        return self.sub_gradient(x_d) + x_d / delta


class NullSparsityEnforcer(SparsityEnforcer):
    def forward(self, x: torch.Tensor):
        return 0.0

    def proximity_operator(self, x: torch.Tensor, delta: float = 1.0):
        return x


class L1SparsityEnforcer(SparsityEnforcer):
    def forward(self, x: torch.Tensor):
        return self.weight * torch.norm(x, p=1)

    def proximity_operator(self, x: torch.Tensor, delta: float = 1.0):
        return torch.sign(x) * torch.clamp(x.abs() - self.weight, min=0.0)

    def sub_gradient(self, v: torch.Tensor):
        return self.weight * torch.sign(v)


class L1L2SparsityEnforcer(SparsityEnforcer):
    def forward(self, x: torch.Tensor):

        ### Consistent with official implemenation
        # w = x.shape[-1] ** .5

        # norm = torch.norm(x, p=2, dim=1)
        # norm = torch.norm(norm, p=1)

        ### Math consistent with paper
        ng = x.numel()
        w = ng**0.5

        norm = torch.linalg.vector_norm(x, ord=2)

        return self.weight * norm * w

    def proximity_operator(self, x: torch.Tensor, delta: float = 1.0):
        return torch.sign(x) * torch.clamp(x.abs() - self.weight, min=0.0)

    def sub_gradient(self, v: torch.Tensor):
        ng = v.numel()
        w = ng**0.5

        numerator = self.weight * w

        norm = torch.linalg.vector_norm(v, ord=2)
        indx = torch.where(norm != 0)[0]

        grad = v.clone()

        grad[indx] = v[indx] * numerator / norm[indx]

        return grad


if __name__ == "__main__":
    A = reg_l1_l2(lamda=1.0)
    B = L1L2SparsityEnforcer(weight=1.0)

    N, C = 32, 10
    x = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    x = torch.randn(N, C)

    a = A(x)
    b = B(x)

    print(f"reg_l1_l2 output: {a}")
    print(f"L1L2SparsityEnforcer output: {b}")
    print(f"Diff (%): {1 - a / b}")
