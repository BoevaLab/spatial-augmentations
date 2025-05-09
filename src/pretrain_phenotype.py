"""
Pretraining Script for Spatial Omics Models.

This script handles the pretraining of models using PyTorch Lightning. It leverages Hydra
for configuration management and supports features such as logging, callbacks, and
hyperparameter optimization.

Main Features:
--------------
- Dynamically instantiates the data module, model, trainer, callbacks, and loggers
  based on the provided Hydra configuration.
- Supports pretraining workflows only (no validation or testing).
- Logs hyperparameters and metrics for tracking experiments.
- Handles checkpointing and restores the best model for continued training.

Usage:
------
Pretrain on CPU:
>>> python src/pretrain_phenotype.py trainer=cpu
Pretrain on GPU:
>>> python src/pretrain_phenotype.py trainer=gpu
Pretrain on MacOS with MPS:
>>> python src/pretrain_phenotype.py trainer=mps
Pretrain on multiple GPUs:
>>> python src/pretrain_phenotype.py trainer=gpu trainer.devices=2
Pretrain model with a specific experiment configuration from configs/experiment/:
>>> python src/pretrain_phenotype.py experiment=experiment_name.yaml
Override any parameter from the command line:
>>> python src/pretrain_phenotype.py trainer.max_epochs=20 data.batch_size=64

Functions:
----------
- pretrain: Handles the core pretraining logic.
- main: Entry point for the script, integrates Hydra for configuration management.
"""

import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

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
def pretrain(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Pretrain the model.

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

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics
    metric_dict = train_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain_phenotype.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for pretraining.

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
    metric_dict, _ = pretrain(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
