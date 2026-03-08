# -*- coding: utf-8 -*-

import torch

from torch import nn

from sparse_bregman.utils import generate_2d_grid

__all__ = ["FoldingDecoder"]


class FoldingDecoder(nn.Module):
    def __init__(
        self,
        code_dimension: int = 512,
        num_grid_points: int = 2025,
        out_features: int = 3,
    ):
        super().__init__()

        grid_features: int = 2

        self.first_folding = nn.Sequential(
            nn.Linear(
                in_features=code_dimension + grid_features,
                out_features=code_dimension + grid_features,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=code_dimension + grid_features,
                out_features=code_dimension + grid_features,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=code_dimension + grid_features,
                out_features=out_features,
            ),
        )

        self.second_folding = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                in_features=code_dimension + out_features,
                out_features=code_dimension + grid_features,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=code_dimension + grid_features,
                out_features=code_dimension + grid_features,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=code_dimension + grid_features,
                out_features=out_features,
            ),
        )

        self.register_buffer("grid", generate_2d_grid(grid_size=num_grid_points))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assert x.size(1) == self.grid.size(
        #     1
        # ), f"Input size ({x.size(1)}) must equal grid size ({self.grid.size(1)})"

        t = torch.cat([x, self.grid.repeat(len(x), 1, 1)], dim=2)
        first_folding = self.first_folding(t)
        
        x_ = torch.cat([x, first_folding], dim=2)
        second_folding = self.second_folding(x_)

        return second_folding


if __name__ == "__main__":
    code_dimension = 512
    num_grid_points = 2025
    grid_features = 3
    out_features = 3

    B, M = 32, 1024

    X = torch.randn(B, num_grid_points, code_dimension)

    model = FoldingDecoder(
        code_dimension=code_dimension,
        out_features=out_features,
        num_grid_points=num_grid_points,
    )
