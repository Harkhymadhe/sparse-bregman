# -*- coding: utf-8 -*-

import os
import torch

from torch.utils.data import DataLoader
from torch import optim

from sparse_bregman.data import (
    NumpyShapeNetPartDataset,
    H5ShapeNetPartDataset,
)
from sparse_bregman.models import FoldingAutoEncoder
from sparse_bregman.loss import ExtendedChamferDistanceLoss
from sparse_bregman.optim import (
    LinearBregmanOptimizer,
    ProximalGDOptimizer,
    AdaBregmanOptimizer,
)
from sparse_bregman.optim.sparsity import NullSparsityEnforcer, L1SparsityEnforcer
from sparse_bregman.utils import point_cloud_collate_function, OptimizerError

from tqdm import tqdm

from rich import print

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Model hyperparameters
    point_cloud_features = 3
    num_points = 2048
    grid_features = 2
    out_features = 3
    code_dimension = 512
    num_neighbours = 16

    B, M, N = 16, 512, 2025

    # Dataset split hyperparameters
    train_fraction = 0.5
    test_fraction = 0.2
    normalize = False

    epochs = 330
    device = "cuda"
    opt = "adam"

    # Available optimizers
    opts = ["adam", "adam_breg", "lin_breg", "proximopt"]

    if opt not in opts:
        raise OptimizerError("opt must be one of ", *opts)

    # Optimizer and sparsity enforcer hyperparameters
    betas = (0.9, 0.999)
    lr = 1e-3
    weight_decay = 1e-6
    momentum = 0.9
    epsilon = 1e-8
    # regularizer = NullSparsityEnforcer()
    regularizer = L1SparsityEnforcer()

    # Initialize datasets
    ds = H5ShapeNetPartDataset(num_points=num_points, normalize=normalize)
    train_ds, test_ds = ds.split(
        train_fraction=train_fraction, test_fraction=test_fraction
    )

    # Initialize dataloaders
    train_dl = DataLoader(
        dataset=train_ds, batch_size=B, collate_fn=point_cloud_collate_function
    )
    test_dl = DataLoader(
        dataset=test_ds, batch_size=B, collate_fn=point_cloud_collate_function
    )

    # Instatiate (Chamfer) loss function

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

    # Parameter counts
    num_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())

    print(
        f"Number of parameters: {num_params:,}\nNumber of trainable parameters: {num_grad_params:,}\n\n"
    )

    # Instantiate optimizer

    if opt == "adam":
        optimizer = AdaBregmanOptimizer(
            model.parameters(),
            lr=lr,
            betas=betas,
            epsilon=epsilon,
            regularizer=regularizer,
        )
    elif opt == "lin_breg":
        optimizer = LinearBregmanOptimizer(
            model.parameters(), lr=lr, momentum=momentum, regularizer=regularizer
        )
    elif opt == "proximopt":
        optimizer = ProximalGDOptimizer(
            model.parameters(), lr=lr, momentum=momentum, regularizer=regularizer
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )

    # Training and eval loop
    iteration = 0
    LEAVE = False

    for epoch in range(1, epochs + 1):
        epoch_train_losses = []
        epoch_test_losses = []

        # Clear optimizer gradients
        optimizer.zero_grad()

        # Set model to train mode
        model.train()

        # Iterate over training data batch
        for batch in tqdm(train_dl, desc="Training Phase", position=1, leave=LEAVE):
            # for batch in iter(train_dl):
            iteration += 1

            # Move batch to device
            X = batch[0].to(device)

            # Inference and loss calculation
            X_hat = model(X)
            loss = loss_fn(X, X_hat)

            # Backpropragation of the loss
            loss.backward()

            # Gradient descent
            optimizer.step()

            # Clear out gradients
            optimizer.zero_grad()

            # Log loss to tracking sequence
            epoch_train_losses.append(loss.item())

            # Save model artefacts intermittently
            if iteration in [5000, 10000, 20000, 40000, 100000, 500000, 4000000]:
                parts = __file__.split(os.sep)
                idx = parts.index("src")
                model_path = __file__.split(os.sep)[1 : idx + 1] + ["artefacts"]

                model_path = os.sep + os.sep.join(model_path)
                model_path = f"{model_path}/iteration_{iteration}.pt"

                with torch.no_grad():
                    state_dict = model.state_dict()
                    torch.save(state_dict, model_path)

        model.eval()

        # Turn off gradients and computational graph
        with torch.no_grad():
            # Run eval loop
            for batch in tqdm(
                test_dl, desc="Evaluation Phase", position=1, leave=LEAVE
            ):
                # for batch in iter(test_dl):
                X = batch[0].to(device)
                X_hat = model(X)
                loss = loss_fn(X, X_hat)

                epoch_test_losses.append(loss.item())

        epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        epoch_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)

        epoch_train_losses.clear()
        epoch_test_losses.clear()

        print(
            f"\nEpoch {epoch} | Train loss: {epoch_train_loss:.4f} | Test loss: {epoch_test_loss:.4f}"
        )

        # Save model to disk
        parts = __file__.split(os.sep)
        idx = parts.index("src")
        model_path = __file__.split(os.sep)[1 : idx + 1] + ["artefacts"]

        model_path = os.sep + os.sep.join(model_path)
        model_path = f"{model_path}/epoch_{epoch}_iteration_{iteration}.pt"

        with torch.no_grad():
            state_dict = model.state_dict()
            torch.save(state_dict, model_path)

    print(X_hat.shape)
    print(
        f"Number of parameters: {num_params:,}\nNumber of trainable parameters: {num_grad_params:,}"
    )
    print(f"Loss: {loss:.3f}")
