from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from torch.nn.functional import cosine_similarity
from torch_geometric.data import Batch, Data
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from src.utils import RankedLogger
from src.utils.clustering_utils import set_leiden_resolution
from src.utils.graph_augmentations_phenotype import get_graph_augmentation
from src.utils.schedulers import MomentumScheduler, WarmupScheduler

log = RankedLogger(__name__, rank_zero_only=True)


class BGRLPhenotypeLitModule(LightningModule):
    """
    A PyTorch Lightning module for training the BGRL model on phenotype data.

    This module implements the training logic for the BGRL model, which is designed for self-supervised
    learning on graph data. It supports both pretraining and finetuning modes, with different loss functions
    and training logic for each mode. The module also supports graph augmentations and momentum-based target
    network updates.

    Key Features:
    - Implements the `training_step` method for pretraining and finetuning.
    - Supports graph augmentations for self-supervised learning.
    - Logs training, validation, and test metrics.
    - Configures optimizers and learning rate schedulers.

    Parameters:
    ----------
    mode : str
        The mode of operation. Options are "pretraining", "finetuning", or "evaluation".
    net : torch.nn.Module
        The BGRL model to train, which includes the online encoder, target encoder, and projector.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training (e.g., Adam, AdamW).
    scheduler : torch.optim.lr_scheduler
        The learning rate scheduler to use for training (e.g., ReduceLROnPlateau, CosineAnnealingLR).
    compile : bool
        Whether to use Torch's `torch.compile` for model compilation (requires PyTorch 2.0+).
    augmentation_mode : str
        The graph augmentation mode to use. Options are "baseline" or "advanced".
    augmentation_list1 : List[str]
        List of graph augmentation methods to apply for the first view.
    augmentation_list2 : List[str]
        List of graph augmentation methods to apply for the second view.
    mm : int
        Initial momentum for the moving average update of the target encoder.
    warmup_steps : int
        Number of warmup steps for the momentum scheduler.
    total_steps : int
        Total number of training steps (iterations).
    drop_edge_p1 : float
        Dropout probability for edges in the first augmented view of the graph.
    drop_edge_p2 : float
        Dropout probability for edges in the second augmented view of the graph.
    drop_feat_p1 : float
        Dropout probability for node features in the first augmented view of the graph.
    drop_feat_p2 : float
        Dropout probability for node features in the second augmented view of the graph.
    seed : int
        Random seed for reproducibility.
    pretrained_model_path : str, optional
        Path to a pretrained model for finetuning. Default is None.
    """

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
        ckpt_file: str = None,
    ) -> None:
        """
        Initialize the BGRLPhenotypeLitModule.

        Parameters:
        ----------
        mode : str
            The mode of operation. Options are "pretraining", "finetuning", or "evaluation".
        net : torch.nn.Module
            The BGRL model to train, which includes the online encoder, target encoder, and projector.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training (e.g., Adam, AdamW).
        scheduler : torch.optim.lr_scheduler
            The learning rate scheduler to use for training (e.g., ReduceLROnPlateau, CosineAnnealingLR).
        compile : bool
            Whether to use Torch's `torch.compile` for model compilation (requires PyTorch 2.0+).
        augmentation_mode : str
            The graph augmentation mode to use. Options are "baseline" or "advanced".
        augmentation_list1 : List[str]
            List of graph augmentation methods to apply for the first view.
        augmentation_list2 : List[str]
            List of graph augmentation methods to apply for the second view.
        mm : float
            Initial momentum for the moving average update of the target encoder (value between 0 and 1).
        warmup_steps : int
            Number of warmup steps for the momentum scheduler.
        total_steps : int
            Total number of training steps (iterations).
        drop_edge_p1 : float
            Dropout probability for edges in the first augmented view of the graph.
        drop_edge_p2 : float
            Dropout probability for edges in the second augmented view of the graph.
        drop_feat_p1 : float
            Dropout probability for node features in the first augmented view of the graph.
        drop_feat_p2 : float
            Dropout probability for node features in the second augmented view of the graph.
        seed : int
            Random seed for reproducibility.
        ckpt_file : str, optional
            Path to a pretrained model for finetuning. Default is None.

        Raises:
        -------
        ValueError
            If the mode is not one of "pretraining", "finetuning", or "evaluation".
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.mode = mode

        # initialize the model (either BGRL for pretraining or GNN_pred for finetuning)
        self.net = net
        if self.mode == "finetuning" and ckpt_file is not None:
            net.from_pretrained(ckpt_file)
            log.info(f"Loaded pretrained model from {ckpt_file}.")

        # loss function
        if mode == "pretraining":
            self.criterion = self.cosine_similarity_loss
        elif mode == "finetuning":
            self.criterion = self.binary_cross_entropy_loss
        elif mode != "evaluation":
            raise ValueError(
                "Invalid mode. Choose either 'pretraining', 'finetuning', or 'evaluation."
            )

        # loss metrics (only calculated during pretraining or finetuning)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # initialize momentum scheduler
        self.momentum_scheduler = MomentumScheduler(
            base_momentum=1 - mm,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        # validation metrics
        self.val_auroc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()
        self.val_prec = BinaryPrecision()
        self.val_rec = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_confusion_matrix = BinaryConfusionMatrix()
        self.val_outputs = []

        # test metrics (only calculated during testing)
        self.test_auroc = BinaryAUROC()
        self.test_acc = BinaryAccuracy()
        self.test_prec = BinaryPrecision()
        self.test_rec = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_confusion_matrix = BinaryConfusionMatrix()
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
        """
        Perform a forward pass through the BGRL model for pretraining.

        Parameters:
        ----------
        online_x : torch.Tensor
            Input node features for the online encoder.
        online_edge_index : torch.Tensor
            Edge indices for the online encoder.
        online_edge_weight : torch.Tensor
            Edge weights for the online encoder.
        target_x : torch.Tensor
            Input node features for the target encoder.
        target_edge_index : torch.Tensor
            Edge indices for the target encoder.
        target_edge_weight : torch.Tensor
            Edge weights for the target encoder.

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - online_q: The predictions from the online network.
            - target_y: The target embeddings from the target network.
        """
        return self.net(
            online_x,
            online_edge_index,
            online_edge_weight,
            target_x,
            target_edge_index,
            target_edge_weight,
        )

    def forward_gnn_pred(self, data: Data) -> list:
        """
        Perform a forward pass through the GNN model for finetuning.

        Parameters:
        ----------
        data : Data
            A PyTorch Geometric `Data` object containing the graph data.

        Returns:
        -------
        list
            The predictions from the GNN model.
        """
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

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        This method applies graph augmentations, computes the forward pass, calculates the loss,
        and updates the target network (if in pretraining mode).

        Parameters:
        ----------
        batch : Data
            A batch of input data.
        batch_idx : int
            The index of the batch.

        Returns:
        -------
        torch.Tensor
            The training loss for the batch.
        """
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
            res = self.forward_gnn_pred(batch)
            y_pred = res[0].flatten()

            # compute binary cross-entropy loss
            y_true = batch.y
            weight = batch.w
            loss = self.criterion(y_pred, y_true, weight)

        # no other training mode is supported
        else:
            raise ValueError(
                "Invalid mode for training. Choose either 'pretraining' or 'finetuning'."
            )

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

    def calculate_metrics(self, logits, labels):
        """
        Evaluate graphs by computing classification metrics.

        Parameters:
        ----------
        logits : torch.Tensor
            The predicted logits for the graphs.
        labels : torch.Tensor
            The ground truth label for the graphs.

        Returns:
        -------
        dict
            A dictionary containing the computed metrics.
        """
        # turn logits into probabilities and predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()

        # compute metrics
        metrics = {
            "auroc": self.val_auroc(probs, labels).item(),
            "accuracy": self.val_acc(preds, labels).item(),
            "f1": self.val_f1(preds, labels).item(),
            "precision": self.val_prec(preds, labels).item(),
            "recall": self.val_rec(preds, labels).item(),
        }

        # compute balanced accuracy
        cm = self.val_confusion_matrix(preds, labels)
        tn, fp, fn, tp = cm.flatten()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        metrics["balanced_accuracy"] = ((sensitivity + specificity) / 2).item()

        return metrics

    def validation_step(self, batch: list[Data], batch_idx: int) -> None:
        """
        Perform a single validation step.

        This method evaluates the model on a batch of validation data and computes metrics.

        Parameters:
        ----------
        batch : list[Data]
            A batch of input data containing subgraphs.
        batch_idx : int
            The index of the batch.
        """
        # create a batch of all subgraphs from list of subgraphs
        region_id = batch.region_id[0][0]

        # run online encoder to get predictions for graph
        # prediction for graph is the mean of all subgraphs
        with torch.no_grad():
            res = self.net(batch)
        y_pred = res[0].flatten().mean()

        # get ground truth labels (graph level, same for all subgraphs)
        y_true = batch.y[0]

        # save the predictions, ground truth labels, and region_id for the graph
        self.val_outputs.append(
            {
                "y_pred": y_pred,
                "y_true": y_true,
                "region_id": region_id,
            }
        )

        # calculate the loss for the batch
        loss = self.criterion(y_pred, y_true, batch.w[0])
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Aggregate metrics at the end of the validation epoch and log the results.
        """
        # aggregate predictions and labels into tensors
        y_preds = torch.tensor([d["y_pred"] for d in self.val_outputs], dtype=torch.float)
        y_trues = torch.tensor([d["y_true"] for d in self.val_outputs], dtype=torch.float)

        # compute metrics across all graphs
        metrics = self.calculate_metrics(y_preds, y_trues)

        # log the results
        self.log("val/auroc", metrics["auroc"], on_epoch=True, prog_bar=True)
        self.log("val/accuracy", metrics["accuracy"], on_epoch=True, prog_bar=True)
        self.log(
            "val/balanced_accuracy", metrics["balanced_accuracy"], on_epoch=True, prog_bar=True
        )
        self.log("val/f1", metrics["f1"], on_epoch=True, prog_bar=True)
        self.log("val/precision", metrics["precision"], on_epoch=True, prog_bar=True)
        self.log("val/recall", metrics["recall"], on_epoch=True, prog_bar=True)

        # clear the outputs for the next validation run
        self.val_outputs.clear()

    def test_step(self, batch: list[Data], batch_idx: int) -> None:
        """
        Perform a single test step.

        This method evaluates the model on a batch of test data and computes metrics.

        Parameters:
        ----------
        batch : list[Data]
            A batch of input data containing subgraphs.
        batch_idx : int
            The index of the batch.
        """
        # create a batch of all subgraphs from list of subgraphs
        region_id = batch.region_id[0][0]

        # run online encoder to get predictions for graph
        # prediction for graph is the mean of all subgraphs
        with torch.no_grad():
            res = self.net(batch)
        y_pred = res[0].flatten().mean()

        # get ground truth labels (graph level, same for all subgraphs)
        y_true = batch.y[0]

        # save the predictions, ground truth labels, and region_id for the graph
        self.test_outputs.append(
            {
                "y_pred": y_pred,
                "y_true": y_true,
                "region_id": region_id,
            }
        )

    def on_test_epoch_end(self) -> None:
        """
        Aggregate metrics at the end of the test epoch and log the results.
        """
        # aggregate predictions and labels into tensors
        y_preds = torch.tensor([d["y_pred"] for d in self.test_outputs], dtype=torch.float)
        y_trues = torch.tensor([d["y_true"] for d in self.test_outputs], dtype=torch.int)

        # compute metrics across all graphs
        metrics = self.calculate_metrics(y_preds, y_trues)

        # log the results
        self.log("test/auroc", metrics["auroc"], on_epoch=True, prog_bar=True)
        self.log("test/accuracy", metrics["accuracy"], on_epoch=True, prog_bar=True)
        self.log(
            "test/balanced_accuracy", metrics["balanced_accuracy"], on_epoch=True, prog_bar=True
        )
        self.log("test/f1", metrics["f1"], on_epoch=True, prog_bar=True)
        self.log("test/precision", metrics["precision"], on_epoch=True, prog_bar=True)
        self.log("test/recall", metrics["recall"], on_epoch=True, prog_bar=True)

        # clear the outputs for the next test run
        self.test_outputs.clear()

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
