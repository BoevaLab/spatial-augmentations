import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
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


@task_wrapper
def finetune(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Finetune the model and optionally evaluate it on a test set using the best weights
    obtained during finetuning.

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

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_phenotype.yaml")
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

    # pretrain the model
    log.info("Pretraining the model...")
    _, pretrain_object_dict = pretrain(cfg.pretrain)

    # get pretrained model checkpoint and insert into finetune config
    ckpt_path = pretrain_object_dict["trainer"].checkpoint_callback.best_model_path
    if not ckpt_path:
        log.warning("Best ckpt not found! Finetuning will start from scratch...")
    else:
        log.info(f"Using pretrained model checkpoint: {ckpt_path}.")
        cfg.finetune.model.ckpt_file = ckpt_path

    # finetune the model
    log.info("Finetuning the model...")
    finetune_metric_dict, _ = finetune(cfg.finetune)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    # in this case the finetuning metric is the metric to be optimized
    metric_value = get_metric_value(
        metric_dict=finetune_metric_dict, metric_name=cfg.finetune.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
