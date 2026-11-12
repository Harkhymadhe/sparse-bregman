# -*- coding: utf-8 -*-

import torch

from torch import nn, optim
from regularizers import *

__all__ = ["SparsityEnforcer", "L1SparsityEnforcer", "L1L2SparsityEnforcer"]


class SparsityEnforcer(nn.Module):
    def __init__(self, weight: float = 1.0, requires_grad: bool = False):
        super().__init__()

        self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32), requires_grad=requires_grad)
    
    def proximity_operator(self, x):
        raise NotImplementedError("Subclasses must implement the proximity operator.")
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    def sub_gradient(self, v):
        return torch.zeros_like(v)


class L1SparsityEnforcer(SparsityEnforcer):
    def forward(self, x):
        return self.weight * torch.norm(x, p=1)
    
    def proximity_operator(self, x):
        return torch.sign(x) * torch.clamp(x.abs() - self.weight, min=0.0)


class L1L2SparsityEnforcer(SparsityEnforcer):
    def forward(self, x):

        ### Consistent with official implemenation
        # w = x.shape[-1] ** .5

        # norm = torch.norm(x, p=2, dim=1)
        # norm = torch.norm(norm, p=1)

        ### Math consistent with paper
        ng = x.numel()
        w = ng ** .5

        norm = torch.linalg.vector_norm(x, ord=2)
        
        return self.weight * norm * w
    
    def proximity_operator(self, x):
        return torch.sign(x) * torch.clamp(x.abs() - self.weight, min=0.0)


if __name__ == "__main__":

    A = reg_l1_l2(lamda=1.0)
    B = L1L2SparsityEnforcer(weight=1.0)

    x = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])

    a = A(x)
    b = B(x)

    print(f"reg_l1_l2 output: {a}")
    print(f"L1L2SparsityEnforcer output: {b}")