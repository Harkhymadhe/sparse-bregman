# -*- coding: utf-8 -*-

from torch import nn

from sparse_bregman.encoder import FoldingEncoder
from sparse_bregman.decoder import FoldingDecoder

__all__ = ["FoldingAutoEncoder"]


class FoldingAutoEncoder(nn.Module):
    def __init__(
        self,
        point_cloud_features: int = 3,
        code_dimension: int = 512,
        num_neighbours: int = 16,
        num_grid_points: int = 2025,
        grid_features: int = 2,
        out_features: int = 3,
    ):
        super().__init__()

        self.num_grid_points = num_grid_points

        self.encoder = FoldingEncoder(
            point_cloud_features=point_cloud_features,
            code_dimension=code_dimension,
            num_neighbours=num_neighbours,
        )

        self.decoder = FoldingDecoder(
            code_dimension=code_dimension,
            out_features=out_features,
            num_grid_points=num_grid_points,
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z.repeat(1, self.num_grid_points, 1))
