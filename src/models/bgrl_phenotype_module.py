import csv
import os
from typing import Any, Dict, List, Tuple

import scanpy as sc
import torch
import torch_geometric
from lightning import LightningModule
from torch.nn.functional import cosine_similarity
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from src.utils.clustering_utils import set_leiden_resolution
from src.utils.graph_augmentations_phenotype import get_graph_augmentation
from src.utils.schedulers import MomentumScheduler, WarmupScheduler


class BGRLPhenotypeLitModule(LightningModule):

    def __init__(
        self,
        mode: str,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        augmentation_mode: str,
        augmentation_list1: List[str],
        augmentation_list2: List[str],
        mm: int,
        warmup_steps: int,
        total_steps: int,
        drop_edge_p1: float,
        drop_edge_p2: float,
        drop_feat_p1: float,
        drop_feat_p2: float,
        seed: int,
        pretrained_model_path: str = None,
    ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.mode = mode

        # initialize the model (either BGRL for pretraining or GNN_pred for finetuning)
        self.net = net
        if self.mode == "finetuning" and pretrained_model_path is not None:
            net.from_pretrained(pretrained_model_path)

        # loss function
        if mode == "pretraining":
            self.criterion = self.cosine_similarity_loss
        elif mode == "finetuning":
            self.criterion = self.binary_cross_entropy_loss
        else:
            raise ValueError("Invalid mode. Choose either 'pretraining' or 'finetuning'.")

        # loss metrics (only calculated during training)
        self.train_loss = MeanMetric()

        # initialize momentum scheduler
        self.momentum_scheduler = MomentumScheduler(
            base_momentum=1 - mm,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        # validation metrics
        self.val_acc = BinaryAccuracy()
        self.val_prec = BinaryPrecision()
        self.val_rec = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_outputs = []

        # test metrics (only calculated during testing)
        self.test_acc = BinaryAccuracy()
        self.test_prec = BinaryPrecision()
        self.test_rec = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_outputs = []

    def forward_bgrl(
        self,
        online_x: torch.Tensor,
        online_edge_index: torch.Tensor,
        online_edge_weight: torch.Tensor,
        target_x: torch.Tensor,
        target_edge_index: torch.Tensor,
        target_edge_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.net(
            online_x,
            online_edge_index,
            online_edge_weight,
            target_x,
            target_edge_index,
            target_edge_weight,
        )

    def forward_gnn_pred(self, data: torch_geometric.data.Data) -> list:

        return self.net(data)

    def cosine_similarity_loss(self, online_q1, target_y2, online_q2, target_y1):
        """
        Compute the cosine similarity loss for the BGRL method.

        Parameters:
        ----------
        online_q1 : torch.Tensor
            Predictions from the online network for the first view.
        target_y2 : torch.Tensor
            Target embeddings for the second view.
        online_q2 : torch.Tensor
            Predictions from the online network for the second view.
        target_y1 : torch.Tensor
            Target embeddings for the first view.

        Returns:
        -------
        torch.Tensor
            The cosine similarity loss.
        """
        loss = (
            2
            - cosine_similarity(online_q1, target_y2.detach(), dim=-1).mean()
            - cosine_similarity(online_q2, target_y1.detach(), dim=-1).mean()
        )
        return loss

    def binary_cross_entropy_loss(self, y_pred, y_true, weight, mean=True):
        """
        Compute the binary cross-entropy loss.

        Parameters:
        ----------
        y_pred : torch.Tensor
            Predictions from the model.
        y_true : torch.Tensor
            Ground truth labels.
        weight : torch.Tensor
            Weights for each sample.
        mean : bool, optional
            If True, return the mean loss. Default is True.

        Returns:
        -------
        torch.Tensor
            The binary cross-entropy loss.
        """
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            y_pred, y_true, weight=weight, reduction="none"
        )
        if mean:
            return bce_loss.mean()
        else:
            return bce_loss.sum()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        # forward pass and loss calculation for pretraining
        if self.mode == "pretraining":

            # get augmented views
            transform1 = get_graph_augmentation(
                self.hparams.augmentation_mode,
                self.hparams.augmentation_list1,
                self.hparams.drop_edge_p1,
                self.hparams.drop_feat_p1,
            )
            transform2 = get_graph_augmentation(
                self.hparams.augmentation_mode,
                self.hparams.augmentation_list2,
                self.hparams.drop_edge_p2,
                self.hparams.drop_feat_p2,
            )

            augmented1 = transform1(batch)
            augmented2 = transform2(batch)

            # forward pass
            q1, y2 = self.forward_bgrl(
                augmented1.x,
                augmented1.edge_index,
                augmented1.edge_weight,
                augmented2.x,
                augmented2.edge_index,
                augmented2.edge_weight,
            )
            q2, y1 = self.forward_bgrl(
                augmented2.x,
                augmented2.edge_index,
                augmented2.edge_weight,
                augmented1.x,
                augmented1.edge_index,
                augmented1.edge_weight,
            )

            # compute cosine similarity loss
            loss = self.criterion(q1, y2, q2, y1)

        # forward pass and loss calculation for finetuning
        elif self.mode == "finetuning":

            # forward pass
            y_pred = self.forward_gnn_pred(batch)

            # compute binary cross-entropy loss
            y_true = batch.y
            weight = batch.weight
            loss = self.criterion(y_pred, y_true, weight)

        # no other mode is supported
        else:
            raise ValueError("Invalid mode. Choose either 'pretraining' or 'finetuning'.")

        # log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        # update target network in pretraining mode
        if self.mode == "pretraining":
            current_step = self.trainer.global_step
            mm = 1 - self.momentum_scheduler.get(current_step)
            self.net.update_target_network(mm)

        return loss

    def setup(self, stage: str) -> None:
        """
        Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Parameters:
        ----------
        stage : str
            Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
        -------
        Dict[str, Any]
            A dict containing the configured optimizers and learning-rate schedulers.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            base_lr = optimizer.param_groups[0]["lr"]
            after_scheduler = self.hparams.scheduler(optimizer=optimizer)
            scheduler = WarmupScheduler(
                optimizer=optimizer,
                base_lr=base_lr,
                warmup_steps=self.hparams.warmup_steps,
                after_scheduler=after_scheduler,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = BGRLPhenotypeLitModule(None, None, None, None)
