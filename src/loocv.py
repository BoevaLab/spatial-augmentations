"""
Training Script for Spatial Omics Models.

This script handles the training and evaluation of models using PyTorch Lightning. It leverages Hydra for configuration 
management and supports features such as logging, callbacks, and hyperparameter optimization.

Main Features:
--------------
- Instantiates the data module, model, trainer, callbacks, and loggers based on the provided Hydra configuration.
- Supports training, validation, and testing workflows.
- Logs hyperparameters and metrics for tracking experiments.
- Handles checkpointing and restores the best model for testing.

Usage:
------
Train on CPU:
>>> python src/train.py trainer=cpu
Train on GPU:
>>> python src/train.py trainer=gpu
Train on MacOS with MPS:
>>> python src/train.py trainer=mps
Train on multiple GPUs:
>>> python src/train.py trainer=gpu trainer.devices=2
Train model with chosen experiment configuration from configs/experiment/:
>>> python src/train.py experiment=experiment_name.yaml
You can override any parameter from command line like this:
>>> python src/train.py trainer.max_epochs=20 data.batch_size=64

Functions:
----------
- train: Handles the core training and testing logic.
- main: Entry point for the script, integrates Hydra for configuration management.
"""

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
import os
import sys
import csv
from sklearn.model_selection import LeaveOneOut
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch_geometric.data import DataLoader

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def loocv(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train the model and optionally evaluate it on a test set using the best weights 
    obtained during training.

    Parameters:
    ----------
    cfg : DictConfig
        A configuration object composed by Hydra.

    Returns:
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        A tuple containing metrics and a dictionary with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    datamodule.prepare_data()
    datamodule.setup()
    dataset = datamodule.dataset

    loo = LeaveOneOut()
    all_metrics = []

    for train_idx, test_idx in loo.split(dataset):
        log.info(f"LOOCV iteration: Train indices {train_idx}, Test index {test_idx}")

        # Split the dataset
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)

        # Update datamodule with subsets
        datamodule.train_dataloader = lambda: DataLoader(train_subset, batch_size=cfg.data.batch_size)
        datamodule.test_dataloader = lambda: DataLoader(test_subset, batch_size=1)

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        log.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("Logging hyperparameters!")
            log_hyperparameters(object_dict)

        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        train_metrics = trainer.callback_metrics

        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)

        test_metrics = trainer.callback_metrics

        # merge train and test metrics
        metric_dict = {**train_metrics, **test_metrics}

        all_metrics.append(metric_dict)

    aggregated_metrics = {key: torch.mean(torch.tensor([m[key] for m in all_metrics])) for key in all_metrics[0]}
    return aggregated_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="loocv.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for training.

    Parameters:
    ----------
    cfg : DictConfig
        A configuration object composed by Hydra.

    Returns:
    -------
    Optional[float]
        The optimized metric value, if applicable.
    """
    # log the command-line arguments
    command = " ".join(sys.argv)
    log.info(f"Command: python {command}")
    
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = loocv(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # save the aggregated metrics to a CSV file
    output_csv_path = os.path.join(cfg.paths.output_dir, "aggregated_metrics.csv")
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Metric", "Value"])
        for key, value in metric_dict.items():
            writer.writerow([key, value])
    log.info(f"Aggregated metrics saved to {output_csv_path}.")

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
