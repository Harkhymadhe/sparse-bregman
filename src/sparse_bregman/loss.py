# -*- coding: utf-8 -*-

import torch
from torch import nn

__all__ = ["ExtendedChamferDistanceLoss"]


class ExtendedChamferDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        Compute the Chamfer distance between two point clouds x and y.

        Args:
            x (torch.Tensor): First point cloud of shape (batch_size, num_points_x, num_dims).
            y (torch.Tensor): Second point cloud of shape (batch_size, num_points_y, num_dims).

        Returns:
            torch.Tensor: Chamfer distance of shape (batch_size,).
        """
        bs, num_points_x, points_dim = input.size()
        _, num_points_y, _ = target.size()

        xx = torch.bmm(input, input.transpose(2, 1))
        yy = torch.bmm(target, target.transpose(2, 1))
        xy = torch.bmm(input, target.transpose(2, 1))

        # rx = xx.diag().unsqueeze(1).expand_as(xy)
        # ry = yy.diag().unsqueeze(0).expand_as(xy)
        diag_func = torch.vmap(torch.diag, in_dims=0)

        rx = diag_func(xx).unsqueeze(-1).expand_as(xy)
        ry = diag_func(yy).unsqueeze(1).expand_as(xy)

        P = rx + ry - 2 * xy

        min_x_to_y, min_y_to_x = P.min(2).values.mean(-1), P.min(1).values.mean(-1)

        distances = torch.cat(
            [min_x_to_y.unsqueeze(-1), min_y_to_x.unsqueeze(-1)], dim=-1
        )
        final_loss = distances.max(dim=-1).values.sum()

        return final_loss


def chamfer_distance(x, y):
    """
    Compute the Chamfer distance between two point clouds x and y.

    Args:
        x (torch.Tensor): First point cloud of shape (batch_size, num_points_x, num_dims).
        y (torch.Tensor): Second point cloud of shape (batch_size, num_points_y, num_dims).

    Returns:
        torch.Tensor: Chamfer distance of shape (batch_size,).
    """
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()

    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    xy = torch.bmm(x, y.transpose(2, 1))

    # rx = xx.diag().unsqueeze(1).expand_as(xy)
    # ry = yy.diag().unsqueeze(0).expand_as(xy)
    diag_func = torch.vmap(torch.diag, in_dims=0)
    rx = diag_func(xx).unsqueeze(-1).expand_as(xy)
    ry = diag_func(yy).unsqueeze(1).expand_as(xy)

    P = rx + ry - 2 * xy
    min_x_to_y, min_y_to_x = P.min(2).values.mean(-1), P.min(1).values.mean(-1)

    A = torch.cat([min_x_to_y.unsqueeze(-1), min_y_to_x.unsqueeze(-1)], dim=-1)
    Amax = A.max(dim=-1).values
    final_loss = Amax.mean()

    print(final_loss)

    print("A: ", A.shape)
    print("Amax: ", Amax.shape)
    print(min_x_to_y.shape)

    print(P.shape)
    print(min_x_to_y.shape)
    print(min_y_to_x.shape)

    dist_x_to_y = P.min(2)[0].mean(1)
    dist_y_to_x = P.min(1)[0].mean(1)

    return dist_x_to_y + dist_y_to_x


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    num_points_x = 10
    num_points_y = 15
    num_dims = 3

    x = torch.randn(batch_size, num_points_x, num_dims)
    y = torch.randn(batch_size, num_points_y, num_dims)

    cd = chamfer_distance(x, y)
    print(cd)
