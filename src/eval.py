"""
Evaluation Script for Spatial Omics Models.

This script handles the evaluation of pre-trained models using PyTorch Lightning. It leverages Hydra for configuration 
management and supports features such as logging, hyperparameter tracking, and flexible checkpoint loading.

Main Features:
--------------
- Instantiates the data module, model, trainer, and loggers based on the provided Hydra configuration.
- Evaluates the model on the test dataset using a specified checkpoint.
- Logs hyperparameters and metrics for experiment tracking.
- Supports additional utilities such as crash handling and configuration tree printing.

Usage:
------
Evaluate on CPU:
>>> python src/eval.py trainer=cpu ckpt_path=/path/to/checkpoint.ckpt
Evaluate on GPU:
>>> python src/eval.py trainer=gpu ckpt_path=/path/to/checkpoint.ckpt
Evaluate on MacOS with MPS:
>>> python src/eval.py trainer=mps ckpt_path=/path/to/checkpoint.ckpt
Evaluate on multiple GPUs:
>>> python src/eval.py trainer=gpu trainer.devices=2 ckpt_path=/path/to/checkpoint.ckpt
Evaluate model with chosen experiment configuration from configs/experiment/:
>>> python src/eval.py experiment=experiment_name.yaml ckpt_path=/path/to/checkpoint.ckpt
You can override any parameter from the command line like this:
>>> python src/eval.py trainer.max_epochs=20 data.batch_size=64 ckpt_path=/path/to/checkpoint.ckpt

Functions:
----------
- evaluate: Handles the core evaluation logic, including model and datamodule instantiation.
- main: Entry point for the script, integrates Hydra for configuration management.
"""

from typing import Any, Dict, List, Tuple

import hydra
import sys
import rootutils
import lightning as L
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a given checkpoint on a datamodule test set.

    This method is wrapped in an optional @task_wrapper decorator, which controls behavior during
    failure. It is useful for multiruns, saving information about crashes, etc.

    Parameters:
    ----------
    cfg : DictConfig
        A configuration object composed by Hydra.

    Returns:
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        A tuple containing metrics and a dictionary with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for evaluation.

    This function applies extra utilities (e.g., asking for tags if none are provided in the configuration,
    printing the configuration tree, etc.) and invokes the evaluation process.

    Parameters:
    ----------
    cfg : DictConfig
        A configuration object composed by Hydra.
    """
    # log the command-line arguments
    command = " ".join(sys.argv)
    log.info(f"Command: python {command}")

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
