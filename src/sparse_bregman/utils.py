# -*- coding: utf-8 -*-

import json
import torch

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from torch import Tensor
from torch_geometric.data import Data

__all__ = [
    "_create_graph",
    "_create_batch_graph",
    "_local_covariance_knn",
    "local_covariance_knn",
    "generate_2d_grid",
    "point_cloud_collate_function",
]


def _create_graph(data: Tensor, neighbours: Tensor, label: int = 0):
    from_edges = np.array(
        [[i for i in range(neighbours.shape[1])] for _ in range(neighbours.shape[0])]
    ).flatten()
    neighbours = np.concatenate(
        [from_edges.reshape(1, -1), neighbours.flatten().reshape(1, -1)], axis=0
    )
    neighbours = np.concatenate([neighbours, neighbours[::-1, :]], axis=1)

    data = Data(
        x=data,
        edge_index=torch.tensor(neighbours).to(torch.int32),
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


@torch.no_grad()
def generate_2d_grid(grid_size: int = 2025, limit: float = 3.0):
    grid_side_size = int(grid_size**0.5)

    # start, stop = (-grid_side_size / 2, grid_side_size / 2)
    # Todo: fix above line of code to use the below instead. Makes for a more granular grid. As it stands, grid has low granularity, meaning folding operation has more work to do to cut/squash the 2D grid into 3D point cloud.
    start, stop = (-limit, limit)

    edge = torch.linspace(start, stop, grid_side_size)

    return torch.cartesian_prod(edge, edge).unsqueeze(0)


# def point_cloud_collate_function(samples):
#     def pad_tensor(t, T):
#         T[: t.size(0), :] = t
#         return T.unsqueeze(0)

#     # pad_fn = torch.vmap(pad_tensor)

#     max_cloud_size = max(len(c[0]) for c in samples)
#     template_tensor = torch.zeros(
#         max_cloud_size, samples[0][0].size(1), dtype=torch.float32
#     )

#     batch_tensor = torch.cat(
#         [pad_tensor(t=sample[0], T=template_tensor) for sample in samples], dim=0
#     )
#     batch_label = torch.cat([sample[1].view(1, 1) for sample in samples], dim=0)

#     return batch_tensor, batch_label


def point_cloud_collate_function(samples):
    batch_tensor = torch.cat([sample[0].unsqueeze(0) for sample in samples], dim=0)
    batch_label = torch.cat([sample[1].view(1, 1) for sample in samples], dim=0)

    return batch_tensor, batch_label


def display_point_cloud(point_cloud, title: str = "Image Title"):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)

    plt.show()
    return


def load_state_dict(path: str, device: str = "cpu"):
    state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in torch.load(path, map_location=device).items()
    }
    return state_dict


def load_json_label_file(fpath: str):
    with open(fpath, "r") as f:
        labels = json.load(f)

    return labels
