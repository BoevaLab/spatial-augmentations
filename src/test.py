import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch_geometric.data import Batch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.bgrl_phenotype_module import BGRLPhenotypeLitModule
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_phenotype.yaml")
def main(cfg: DictConfig):

    cfg.pretrain.data.mode = "pretraining"
    cfg.pretrain.data.num_regions_per_segment = 2
    cfg.pretrain.data.steps_per_segment = 20

    cfg.finetune.data.mode = "finetuning"
    cfg.finetune.data.redo_preprocess = False
    cfg.finetune.data.num_regions_per_segment = 2
    cfg.finetune.data.steps_per_segment = 100

    cfg.task_name = "test"

    cfg.pretrain.model.augmentation_mode = "baseline"
    cfg.pretrain.model.augmentation_list1 = ["DropFeatures", "DropEdges"]
    cfg.pretrain.model.augmentation_list2 = ["DropFeatures", "DropEdges"]

    cfg.pretrain.trainer.min_epochs = 1
    cfg.pretrain.trainer.max_epochs = 100
    cfg.pretrain.trainer.log_every_n_steps = 10
    cfg.pretrain.trainer.check_val_every_n_epoch = 99999
    cfg.pretrain.trainer.limit_train_batches = 1

    cfg.finetune.trainer.min_epochs = 1
    cfg.finetune.trainer.max_epochs = 50000
    cfg.finetune.trainer.log_every_n_steps = 10
    cfg.finetune.trainer.check_val_every_n_epoch = 1000
    cfg.finetune.trainer.limit_train_batches = 1

    extras(cfg)

    print(f"Instantiating datamodule <{cfg.finetune.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.finetune.data)
    datamodule.setup()

    cfg.finetune.model.ckpt_file = "logs/epoch_epoch=8021.ckpt"

    print(f"Instantiating model <{cfg.finetune.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.finetune.model)

    log.info(f"Instantiating trainer <{cfg.finetune.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.finetune.trainer)

    if cfg.pretrain.get("train"):
        print("Starting training!")
        # trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    trainer.validate(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
