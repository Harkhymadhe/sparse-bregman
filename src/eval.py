# -*- coding: utf-8 -*-

import os

import numpy as np
import torch

from sparse_bregman.utils import display_point_cloud, load_json_label_file

from sparse_bregman.data import ShapeNetPartDataset, ShapeNetPartDatasetV2
from sparse_bregman.models import FoldingAutoEncoder
from sparse_bregman.utils import display_point_cloud, load_state_dict

if __name__ == "__main__":
    point_cloud_features = 3
    num_points = 2048
    grid_features = 2
    out_features = 3
    code_dimension = 512
    num_neighbours = 16

    B, M, N = 16, 512, 2025
    train_fraction = .5

    normalize = False

    device="cpu"

    model = FoldingAutoEncoder(
        point_cloud_features=point_cloud_features,
        code_dimension=code_dimension,
        num_neighbours=num_neighbours,
        grid_features=grid_features,
        out_features=out_features,
        num_grid_points=N,
    )

    # dataset = ShapeNetPartDataset(num_points=num_points, normalize=normalize)
    dataset = ShapeNetPartDatasetV2(num_points=num_points, normalize=normalize)
    
    if isinstance(dataset, ShapeNetPartDataset):
        train_dataset = ShapeNetPartDataset(num_points=num_points, normalize=normalize).split(train_fraction = train_fraction)[0]
    else:
        train_dataset = dataset

    epoch = 330
    iteration = epoch * int(len(train_dataset)//B + 1)
    num_samples = 3
    category = 15

    parts = __file__.split(os.sep)
    idx = parts.index("src")
    model_path = __file__.split(os.sep)[1:idx+1] + ["artefacts"]
    
    model_path = os.sep + os.sep.join(model_path)
    model_path = f"{model_path}/epoch_{epoch}_iteration_{iteration}.pt"

    state_dict = load_state_dict(path=model_path, device=device)

    model.load_state_dict(state_dict)
    model.eval()

    # print(f"Decoder parameter count: {sum(p.numel() for p in model.decoder.parameters()):,}")

    samples = dataset.sample(num_samples = num_samples, category = category)

    Xs = [s[0].unsqueeze(0) for s in samples]
    ys = [dataset.index2label[s[1].item()] for s in samples]

    # rand_index = np.random.randint(low=0, high=len(dataset))

    print(dataset.category_counts(normalize=normalize))

    # X, y = dataset[rand_index]

    X = torch.cat(Xs, dim=0)
    # y = dataset.index2label[y.item()]

    X_hat = model(X).detach()

    for i in range(num_samples):
        x = Xs[i]
        x_hat = X_hat[i]
        y = ys[i]

        display_point_cloud(x.squeeze(0), title = f"Actual Point Cloud: {y}")
        display_point_cloud(x_hat.squeeze(0), title = f"Reconstructed Point Cloud: {y}")
