# -*- coding: utf-8 -*-

import numpy as np
import torch

from torch_geometric import nn as gnn
from torch_geometric.data import Data, Batch

from torch_geometric.nn import GCNConv

from torch import nn, Tensor
from torch.nn import functional as F

__all__ = [
    "_create_graph",
    "_create_batch_graph",
    "_local_covariance_knn",
    "local_covariance_knn",
    "GraphMLP",
    "FoldingEncoder",
]


def _create_graph(data: Tensor, neighbours: Tensor, label: int = 0):
    from_edges = (
        torch.tensor(
            [
                [i for i in range(neighbours.shape[1])]
                for _ in range(neighbours.shape[0])
            ]
        )
        .flatten()
        .to(data.device)
    )
    neighbours = torch.cat(
        [from_edges.view(1, -1), neighbours.flatten().view(1, -1)], dim=0
    )
    neighbours = torch.cat([neighbours, torch.flip(neighbours, [0])], dim=1)

    data = Data(
        x=data,
        edge_index=neighbours,
        y=torch.tensor([label]),
    )
    return data


def _create_batch_graph(
    batch_data: Tensor, batch_neighbours: Tensor, batch_label: int = 0
):
    xs = batch_data.chunk(chunks=len(batch_data), dim=0)
    neighbours = batch_neighbours.chunk(chunks=len(batch_data), dim=0)

    data_list = [
        _create_graph(data=x.squeeze(0), neighbours=n.squeeze(0))
        for (x, n) in zip(xs, neighbours)
    ]

    return Batch.from_data_list(data_list=data_list)


def _local_covariance_knn(xs, idx, k=5):
    covariances = []
    for x, n in zip(xs, idx):
        # print(x.shape, n.shape)
        # for i in range(len(xs)):
        neighbors = x.squeeze(0)[n.squeeze(0)]  # (k, D)
        neighbors = neighbors - neighbors.mean(0, keepdim=True)
        cov = neighbors.mT @ neighbors / (k - 1)
        # print("Small COV: ", cov.shape)
        covariances.append(cov)

    # print("COV len: ", len(covariances))
    covariances = torch.stack(covariances, dim=0).flatten(start_dim=-2)
    # print("COV: ", covariances.shape)
    return covariances


def local_covariance_knn(X, k=5):
    """
    X: (N, D)
    Returns: (N, D, D) local covariance per sample
    """
    shape = X.shape
    N, D = shape[-2], shape[-1]

    # pairwise distances
    dist = torch.cdist(X, X)

    # indices of k nearest neighbors
    knn_idx = dist.topk(k=k, largest=False).indices[:,]

    xs = X.chunk(chunks=len(X), dim=0)
    idx = knn_idx.chunk(chunks=len(knn_idx), dim=0)

    covariances = _local_covariance_knn(xs=xs, idx=idx, k=k)

    return knn_idx, covariances


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
