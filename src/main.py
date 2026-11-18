# -*- coding: utf-8 -*-

import os

import hydra
import nvtx
import torch
from loguru import logger
from omegaconf import DictConfig
from torch import optim
from torch.cuda import cudart
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparse_bregman.data import H5ShapeNetPartDataset
from sparse_bregman.loss import ExtendedChamferDistanceLoss
from sparse_bregman.models import FoldingAutoEncoder
from sparse_bregman.optim import (
    AdaBregmanOptimizer,
    LinearBregmanOptimizer,
    ProximalGDOptimizer,
)
from sparse_bregman.optim.sparsity import L1SparsityEnforcer
from sparse_bregman.utils import OptimizerError, point_cloud_collate_function

__all__ = ["main"]


@hydra.main(config_name="config", config_path="conf", version_base="1.4")
def main(cfg: DictConfig):
    # Reduce precision for faster computation
    torch.set_float32_matmul_precision("high")

    # Instantiate CUDA profiler
    cuda_profiler = cudart()

    # Add logger from Loguru
    logger.add(sink=cfg.logs.path, enqueue=True)

    # Model hyperparameters
    point_cloud_features = cfg.model.point_cloud_features
    num_grid_points = cfg.model.num_grid_points
    grid_features = cfg.model.grid_features
    out_features = cfg.model.out_features
    code_dimension = cfg.model.code_dimension
    num_neighbours = cfg.model.num_neighbours

    # Dataset hyperparameters
    cache_size = cfg.data.cache_size
    normalize = cfg.data.normalize
    num_points = cfg.data.num_points

    # Dataset split hyperparameters
    train_fraction = cfg.data.train_fraction
    test_fraction = cfg.data.test_fraction

    if test_fraction is None:
        test_fraction = 0.2

    # Artefact persistence
    model_base_path = cfg.artefacts.path

    # Training hyperparameters
    batch_size = cfg.training.data.batch_size
    epochs = cfg.training.num_epochs
    device = torch.device(
        "cpu" if not torch.cuda.is_available() else cfg.training.device
    )
    optimizer_id = cfg.training.optimizer.name

    # Available optimizers
    opts = ["adam", "adam_breg", "lin_breg", "proximopt"]

    if optimizer_id not in opts:
        raise OptimizerError("opt must be one of ", *opts)

    # Optimizer and sparsity enforcer hyperparameters
    betas = (cfg.training.optimizer.beta1, cfg.training.optimizer.beta2)
    lr = cfg.training.optimizer.lr
    weight_decay = cfg.training.optimizer.weight_decay
    momentum = cfg.training.optimizer.momentum
    epsilon = cfg.training.optimizer.epsilon

    delta = cfg.training.regularizer.delta
    weight = cfg.training.regularizer.weight
    requires_grad = cfg.training.regularizer.requires_grad

    # regularizer = NullSparsityEnforcer(requires_grad=requires_grad, weight=weight)
    regularizer = L1SparsityEnforcer(requires_grad=requires_grad, weight=weight)

    # Initialize datasets
    ds = H5ShapeNetPartDataset(
        path=cfg.data.path,
        num_points=num_points,
        normalize=normalize,
        cache_size=cache_size,
    )
    train_ds, test_ds = ds.split(
        train_fraction=train_fraction, test_fraction=test_fraction
    )

    # Initialize dataloaders
    train_dl = DataLoader(
        dataset=train_ds, batch_size=batch_size, collate_fn=point_cloud_collate_function
    )
    test_dl = DataLoader(
        dataset=test_ds, batch_size=batch_size, collate_fn=point_cloud_collate_function
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
        num_grid_points=num_grid_points,
    ).to(device)

    # Start CUDA profiler
    cuda_profiler.cudaProfilerStart()

    # Compile model
    with nvtx.annotate("torch.compile", color="red"):
        model = torch.compile(model)

    # Parameter counts
    num_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())

    logger.info(
        f"Number of parameters: {num_params:,}\nNumber of trainable parameters: {num_grad_params:,}\n\n"
    )

    # Instantiate optimizer

    if optimizer_id == "adam_breg":
        optimizer = AdaBregmanOptimizer(
            model.parameters(),
            lr=lr,
            betas=betas,
            epsilon=epsilon,
            delta=delta,
            regularizer=regularizer,
        )
    elif optimizer_id == "lin_breg":
        optimizer = LinearBregmanOptimizer(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            delta=delta,
            regularizer=regularizer,
        )
    elif optimizer_id == "proximopt":
        optimizer = ProximalGDOptimizer(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            delta=delta,
            regularizer=regularizer,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
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

        with nvtx.annotate(f"Epoch #{epoch}", color="lightgrey"):
            # Iterate over training data batch
            for batch in tqdm(train_dl, desc="Training Phase", position=1, leave=LEAVE):
                
                iteration += 1

                with nvtx.annotate(f"Batch #{iteration}", color="goldenrod"):
                    with nvtx.annotate("Data to CUDA", color="limegreen"):
                        # Move batch to device
                        X = batch[0].to(device)

                    # Inference and loss calculation
                    with nvtx.annotate("Forward pass", color="darkviolet"):
                        X_hat = model(X)

                    with nvtx.annotate("Loss computation", color="violet"):
                        loss = loss_fn(X, X_hat)

                    # Backpropragation of the loss
                    with nvtx.annotate("Backward pass", color="cornflowerblue"):
                        loss.backward()

                    # Gradient descent
                    with nvtx.annotate("Gradient descent", color="royalblue"):
                        optimizer.step()

                    # Clear out gradients
                    with nvtx.annotate("Zero.grad", color="palegoldenrod"):
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

            # Set model to eval mode
            model.eval()

            # Turn off gradients and computational graph
            with nvtx.annotate("Eval process", color="purple"), torch.no_grad():
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

            logger.info(
                f"\nEpoch {epoch} | Train loss: {epoch_train_loss:.4f} | Test loss: {epoch_test_loss:.4f}"
            )

        # Save model to disk
        model_path = f"{model_base_path}/epoch_{epoch}_iteration_{iteration}.pt"

        with torch.no_grad():
            state_dict = model.state_dict()
            torch.save(state_dict, model_path)

        if epoch == 1:
            # Stop CUDA profiler
            cuda_profiler.cudaProfilerStop()


if __name__ == "__main__":
    main()
