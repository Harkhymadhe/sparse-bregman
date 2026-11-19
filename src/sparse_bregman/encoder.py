# -*- coding: utf-8 -*-

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv

from sparse_bregman.utils import (
    _create_graph,
    _create_batch_graph,
    local_covariance_knn,
)

__all__ = [
    "FoldingEncoder",
    "GraphMLP",
]


class GraphMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer1 = GCNConv(in_channels=in_features, out_channels=out_features)
        self.layer2 = GCNConv(in_channels=out_features, out_channels=out_features)

    def forward(self, x, edge_index):
        x = F.relu(self.layer1(x, edge_index))
        x = F.relu(self.layer2(x, edge_index))
        return x


class FoldingEncoder(nn.Module):
    def __init__(
        self,
        point_cloud_features: int = 3,
        code_dimension: int = 512,
        num_neighbours: int = 16,
    ):
        super().__init__()

        self.code_dimension = code_dimension
        self.point_cloud_features = point_cloud_features
        self.num_neighbours = num_neighbours

        self.first_mlp = nn.Sequential(
            nn.Linear(
                in_features=point_cloud_features + point_cloud_features**2,
                out_features=32,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=32,
                out_features=64,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=64,
            ),
        )

        self.graph_mlp = GraphMLP(in_features=64, out_features=int(code_dimension * 2))

        self.last_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                in_features=int(code_dimension * 2),
                out_features=int(code_dimension * 2),
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=int(code_dimension * 2),
                out_features=code_dimension,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = len(x)

        idx, x_ = local_covariance_knn(x, k=self.num_neighbours)

        x_ = self.first_mlp(torch.cat([x, x_], dim=-1))
        data = _create_batch_graph(batch_data=x_, batch_neighbours=idx)
        x_ = self.graph_mlp(data.x, data.edge_index).view(
            B, -1, self.code_dimension * 2
        )
        x_ = x_.max(dim=1).values.unsqueeze(1)
        x_ = self.last_mlp(x_)

        return x_


if __name__ == "__main__":
    point_cloud_features = 3
    grid_features = 3
    code_dimension = 512

    B, M = 32, 512

    X = torch.randn(B // 4, grid_features)
    X = torch.randn(B, M, point_cloud_features)

    idx, X_ = local_covariance_knn(X)
    g = _create_batch_graph(batch_data=X_, batch_neighbours=idx)

    # idx, X_ = local_covariance_knn(X)
    # print(X_.shape)
    # X_ = torch.cat([X, X_.flatten(start_dim=-2)], dim=-1).unsqueeze(0)
    # print(X_.shape)
    # idx = idx.unsqueeze(0)
    # grid = torch.randn(B, M, grid_features)
    model = FoldingEncoder(
        point_cloud_features=point_cloud_features,
        code_dimension=code_dimension,
    )

    print(model(X).shape)

    # print(X_.shape)
    # print(idx)

    # data = create_graph(data = X_, neighbours = idx)
    # print(data)
