# -*- coding: utf-8 -*-

import torch

from torch.optim import Optimizer
from torch.autograd import grad
from torch import nn

__all__ = ["LinearBregmanOptimizer"]


class LinearBregmanOptimizer(Optimizer):
    def __init__(
        self,
        parameters,
        delta: int = 1,
        weight: float = 1.0,
        l1_norm: bool = True,
        lr: float = 1e-3,
        *args,
        **kwargs,
    ):
        super().__init__(parameters, *args, **kwargs, defaults={"lr": lr})

        self.delta = delta
        self.weight = weight
        self.l1_norm = l1_norm

        total_J = self.J()
        print("Total J: ", total_J)
        J_grad = self.sub_gradients(J=total_J)

        print("J_grad: ", J_grad)

        with torch.no_grad():
            self.v_groups = (
                {
                    "params": [
                        J_grad + v.detach().data.abs() / self.delta
                        for v in group["params"]
                    ]
                }
                for group in self.param_groups
            )

    def sub_gradients(self, J: torch.Tensor):
        J_grad = grad(
            outputs=J,
            inputs=[p for group in self.param_groups for p in group["params"]],
        )
        return J_grad

    def proximity_ops(self, v: torch.Tensor) -> torch.Tensor:
        v_ = v.abs() - self.weight
        return self.delta * torch.sign(v) * torch.where(v_ > 0, v_, 0)

    def _l1_norm_J(self, parameter: torch.Tensor):
        return parameter.abs().sum()

    def _l2_norm_J(self, parameter: torch.Tensor):
        n_g = parameter.numel()
        norm = torch.pow((parameter**2).sum(), 0.5)
        return (n_g**0.5) * norm

    # @torch.no_grad()
    def J(self):
        norm_func = self._l1_norm_J if self.l1_norm else self._l2_norm_J
        total_j = torch.tensor(0.0, requires_grad=True)

        for group in self.param_groups:
            for v in group["params"]:
                # norm = norm_func(v.detach())
                norm = norm_func(v)
                print("Norm: ", norm)
                total_j = total_j + norm

        return self.weight * total_j

    def step(self):
        for p_group, v_group in zip(self.param_groups, self.v_groups):
            for param, v in zip(p_group["params"], v_group):
                # v = param.data / self.delta
                v = v - self.lr * param.grad.data
                param.data = self.proximity_ops(v=v)
                # param.grad = None
        return

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None


if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(in_features=10, out_features=2), nn.Softmax(dim=-1))

    # print(list(model.parameters()))

    optim = LinearBregmanOptimizer(parameters=model.parameters(), lr=1e-3)

    X = torch.randn(32, 10)
    y = torch.randint(low=0, high=2, size=(32, 1))

    y_hat = model(X)

    print(y_hat)

    optim.step()
