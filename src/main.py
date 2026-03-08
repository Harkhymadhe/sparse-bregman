# -*- coding: utf-8 -*-

import os
import torch

from torch.utils.data import DataLoader
from torch import optim

from sparse_bregman.data import ShapeNetPartDataset, ShapeNetPartDatasetV2
from sparse_bregman.models import FoldingAutoEncoder
from sparse_bregman.loss import ExtendedChamferDistanceLoss
from sparse_bregman.utils import point_cloud_collate_function

from tqdm import tqdm

from rich import print

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    point_cloud_features = 3
    num_points=2048
    grid_features = 2
    out_features = 3
    code_dimension = 512
    num_neighbours = 16

    B, M, N = 16, 512, 2025

    train_fraction = .5
    test_fraction = .2
    normalize = False

    epochs = 330
    device = "cuda"

    betas = (.9, .999)
    lr = 1e-4
    weight_decay = 1e-6
    
    ds = ShapeNetPartDatasetV2(num_points=num_points, normalize=normalize)
    train_ds, test_ds = ds.split(train_fraction = train_fraction, test_fraction = test_fraction)

    train_dl = DataLoader(dataset=train_ds, batch_size=B, collate_fn = point_cloud_collate_function)
    test_dl = DataLoader(dataset=test_ds, batch_size=B, collate_fn = point_cloud_collate_function)

    loss_fn = ExtendedChamferDistanceLoss()

    # Instantiate model
    model = FoldingAutoEncoder(
        point_cloud_features=point_cloud_features,
        code_dimension=code_dimension,
        num_neighbours=num_neighbours,
        grid_features=grid_features,
        out_features=out_features,
        num_grid_points=N,
    ).to(device)

    # Compile model
    model = torch.compile(model)

    num_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Number of parameters: {num_params:,}\nNumber of trainable parameters: {num_grad_params:,}\n\n")

    optimizer = optim.AdamW(model.parameters(), lr = lr, betas=betas, weight_decay=weight_decay)
    iteration = 0
    LEAVE=False
    
    for epoch in range(1, epochs + 1):
        epoch_train_losses = []
        epoch_test_losses = []

        optimizer.zero_grad()

        model.train()

        for batch in tqdm(train_dl, desc = "Training Phase", position=1, leave=LEAVE):
        # for batch in iter(train_dl):
            iteration += 1
            X = batch[0].to(device)
            X_hat = model(X)
            loss = loss_fn(X, X_hat)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_losses.append(loss.item())

            if iteration in [5000, 10000, 20000, 40000, 100000, 500000, 4000000]:
                parts = __file__.split(os.sep)
                idx = parts.index("src")
                model_path = __file__.split(os.sep)[1:idx+1] + ["artefacts"]
                
                model_path = os.sep + os.sep.join(model_path)
                model_path = f"{model_path}/iteration_{iteration}.pt"

                with torch.no_grad():
                    state_dict = model.state_dict()
                    torch.save(state_dict, model_path)
        
        model.eval()

        with torch.no_grad():
            for batch in tqdm(test_dl, desc = "Evaluation Phase", position=1, leave=LEAVE):
            # for batch in iter(test_dl):
                X = batch[0].to(device)
                X_hat = model(X)
                loss = loss_fn(X, X_hat)

                epoch_test_losses.append(loss.item())
        
        epoch_train_loss = sum(epoch_train_losses)/len(epoch_train_losses)
        epoch_test_loss = sum(epoch_test_losses)/len(epoch_test_losses)
        
        epoch_train_losses.clear()
        epoch_test_losses.clear()

        print(f"Epoch {epoch} | Train loss: {epoch_train_loss:.4f} | Test loss: {epoch_test_loss:.4f}")

        parts = __file__.split(os.sep)
        idx = parts.index("src")
        model_path = __file__.split(os.sep)[1:idx+1] + ["artefacts"]
        
        model_path = os.sep + os.sep.join(model_path)
        model_path = f"{model_path}/epoch_{epoch}_iteration_{iteration}.pt"

        with torch.no_grad():
            state_dict = model.state_dict()
            torch.save(state_dict, model_path)

    print(X_hat.shape)
    print(f"Number of parameters: {num_params:,}\nNumber of trainable parameters: {num_grad_params:,}")
    print(f"Loss: {loss:.3f}")
