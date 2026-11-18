# -*- coding: utf-8 -*-

import torch

from torch.optim import Optimizer
from torch.autograd import grad
from torch import nn

from sparse_bregman.optim.sparsity import SparsityEnforcer, NullSparsityEnforcer

__all__ = ["LinearBregmanOptimizer", "ProximalGDOptimizer", "AdaBregmanOptimizer"]


class LinearBregmanOptimizer(Optimizer):
    def __init__(
        self,
        parameters,
        lr: float = 1e-3,
        momentum: float = 0.0,
        delta: int = 1,
        regularizer: SparsityEnforcer = NullSparsityEnforcer(),
        *args,
        **kwargs,
    ):
        defaults = {
            "lr": lr,
            "delta": delta,
            "momentum": momentum,
            "regularizer": regularizer,
        }
        super().__init__(parameters, *args, **kwargs, defaults=defaults)

    @torch.no_grad()
    def step(self):
        # Extract each parameter group for processing
        for group in self.param_groups:
            # Extract specific values per group
            delta = group["delta"]
            lr = group["lr"]
            momentum = group["momentum"]
            regularizer = group["regularizer"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]

                # Initialize parameter state if empty
                if len(state) == 0:
                    state["v"] = regularizer.initialize_sub_gradient(param, delta=delta)
                    state["m"] = None

                # Process subgradient with or without momentum
                if momentum > 0.0:
                    m = state["m"]

                    if m is None:
                        m = torch.zeros_like(grad)

                    m.mul_(momentum)
                    m.add_(lr * (1 - momentum) * grad)

                    state["m"] = m
                    state["v"].add_(-m)
                else:
                    state["v"].add_(-lr * grad)

                # Update parameters using proximity operator on subgrad and delta
                param.data = regularizer.proximity_operator(
                    x=delta * state["v"], delta=delta
                )

        return self


class ProximalGDOptimizer(Optimizer):
    def __init__(
        self,
        parameters,
        lr: float = 1e-3,
        momentum: float = 0.0,
        delta: int = 1,
        regularizer: SparsityEnforcer = NullSparsityEnforcer(),
        *args,
        **kwargs,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "delta": delta,
            "regularizer": regularizer,
        }

        super().__init__(parameters, *args, **kwargs, defaults=defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            delta = group["delta"]
            momentum = group["momentum"]
            regularizer = group["regularizer"]

            for param in group["params"]:

                if param.grad is None:
                    continue
                
                grad = param.grad.data

                param.data.add_(-lr * grad)
                param.data = regularizer.proximity_operator(x=param.data, delta=lr)

        return self


class AdaBregmanOptimizer(Optimizer):
    def __init__(
        self,
        parameters,
        lr: float = 1e-3,
        betas: tuple[float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        delta: int = 1,
        regularizer: SparsityEnforcer = NullSparsityEnforcer(),
        *args,
        **kwargs,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "delta": delta,
            "epsilon": epsilon,
            "regularizer": regularizer,
        }

        super().__init__(parameters, *args, **kwargs, defaults=defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            delta = group["delta"]
            betas = group["betas"]
            regularizer = group["regularizer"]
            epsilon = group["epsilon"]

            for param in group["params"]:
                grad = param.grad.data
                state = self.state[param]

                if grad is None:
                    continue

                if len(state) == 0:
                    state["step"] = 1
                    state["m1"] = torch.zeros_like(grad)
                    state["m2"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)

                k = state["step"]
                v = state["v"]

                m1 = state["m1"]
                m2 = state["m2"]

                # Update momentum buffers
                m1.mul_(betas[0]).add_((1 - betas[0]) * grad)

                m2.mul_(betas[1]).add_((1 - betas[1]) * grad**2)

                # Bias correction for buffers
                m1.div_(1 - betas[0] ** k)
                m2.div_(1 - betas[1] ** k)

                # Update subgradient
                v.add_(-lr * m1 / (m2**0.5 + epsilon))

                state["step"] += 1

                # state["m1"] = m1
                # state["m2"] = m2
                # state["v"] = v

                # Update parameters using proximity operator on subgrad and delta
                param.data = regularizer.proximity_operator(x=delta * v.data, delta=delta)

        return self


if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(in_features=10, out_features=2), nn.Softmax(dim=-1))

    # print(list(model.parameters()))

    optim = LinearBregmanOptimizer(parameters=model.parameters(), lr=1e-3)

    X = torch.randn(32, 10)
    y = torch.randint(low=0, high=2, size=(32, 1))

    y_hat = model(X)

    print(y_hat)

    optim.step()
