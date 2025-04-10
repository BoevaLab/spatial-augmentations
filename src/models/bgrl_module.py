import csv
import os
from typing import Any, Dict, List, Tuple

import scanpy as sc
import torch
from lightning import LightningModule
from torch.nn.functional import cosine_similarity
from torchmetrics import MeanMetric
from torchmetrics.clustering import (
    AdjustedRandScore,
    CompletenessScore,
    HomogeneityScore,
    NormalizedMutualInfoScore,
)

from src.utils.clustering_utils import set_leiden_resolution
from src.utils.graph_augmentations import get_graph_augmentation
from src.utils.schedulers import MomentumScheduler, WarmupScheduler

# TODO: add CosineSimilarityScheduler as lr and mm scheduler (including warmup steps and so on)
# TODO: file path to data dir in test step (make as config)


class BGRLLitModule(LightningModule):
    """
    A PyTorch Lightning module for training the BGRL (Bootstrap Graph Representation Learning) model.

    This module implements the training logic for the BGRL model, which is designed for self-supervised
    learning on graph data. It uses an online encoder and a target encoder, with the target encoder
    updated using a momentum-based moving average. The module also supports graph augmentations and
    cosine similarity loss for training.

    Key Features:
    - Implements the `training_step` method for a si
    ngle training step.
    - Supports learning rate and momentum scheduling.
    - Logs training metrics such as loss.
    - Configures optimizers and learning rate schedulers.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        augmentation_mode: str = "baseline",
        mm: int = 0.99,
        spatial_regularization_strength: float = 0.0,
        node_subset_sz: int = 5000,
        drop_edge_p1: float = 0.1,
        drop_edge_p2: float = 0.1,
        drop_feat_p1: float = 0.1,
        drop_feat_p2: float = 0.1,
        mu: float = 0.2,
        p_lambda: float = 0.5,
        processed_dir: str = "data/domain/processed/",
    ) -> None:
        """
        Initialize the BGRLLitModule.

        Parameters:
        ----------
        model : torch.nn.Module
            The BGRL model to train.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training.
        scheduler : torch.optim.lr_scheduler
            The learning rate scheduler to use for training.
        compile : bool
            Whether to use Torch's `torch.compile` for model compilation.
        augmentation_mode : str, optional
            The graph augmentation mode to use. Default is "baseline".
        mm : float, optional
            Momentum for the target network updates. Default is 0.99.
        spatial_regularization_strength : float, optional
            Strength of the spatial regularization term. Default is 0.0.
        node_subset_sz : int, optional
            Size of the node subset for spatial regularization. Default is 5000.
        drop_edge_p1 : float, optional
            Dropout probability for edges in the first view. Default is 0.0.
        drop_edge_p2 : float, optional
            Dropout probability for edges in the second view. Default is 0.0.
        drop_feat_p1 : float, optional
            Dropout probability for features in the first view. Default is 0.0.
        drop_feat_p2 : float, optional
            Dropout probability for features in the second view. Default is 0.0.
        mu : float, optional
            The parameter for the graph augmentation. Default is 0.2.
        p_lambda : float, optional
            The parameter for the graph augmentation. Default is 0.5.
        num_iterations : int, optional
            Total number of training iterations. Default is 1e5.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # initialize the BGRL model
        self.net = net

        # loss function
        self.criterion = self.cosine_similarity_loss

        # loss metrics (only calculated during training)
        self.train_loss = MeanMetric()

        # test metrics (only calculated during testing)
        self.test_nmi = NormalizedMutualInfoScore()
        self.test_ars = AdjustedRandScore()
        self.test_homogeneity = HomogeneityScore()
        self.test_completeness = CompletenessScore()
        self.test_outputs = []

    def forward(
        self,
        online_x: torch.Tensor,
        online_edge_index: torch.Tensor,
        online_edge_weight: torch.Tensor,
        target_x: torch.Tensor,
        target_edge_index: torch.Tensor,
        target_edge_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the BGRL model.

        Parameters:
        ----------
        online_x : torch.Tensor
            Input graph for the online encoder.
        target_x : torch.Tensor
            Input graph for the target encoder.

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

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step.

        This method applies graph augmentations, computes the forward pass, calculates the loss,
        and updates the target network.

        Parameters:
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A batch of input data.
        batch_idx : int
            The index of the batch.

        Returns:
        -------
        torch.Tensor
            The training loss for the batch.
        """
        transform1 = get_graph_augmentation(
            self.hparams.augmentation_mode,
            self.hparams.drop_edge_p1,
            self.hparams.drop_feat_p1,
            self.hparams.mu,
            self.hparams.p_lambda,
        )
        transform2 = get_graph_augmentation(
            self.hparams.augmentation_mode,
            self.hparams.drop_edge_p2,
            self.hparams.drop_feat_p2,
            self.hparams.mu,
            self.hparams.p_lambda,
        )

        augmented1 = transform1(batch)
        augmented2 = transform2(batch)

        # forward pass
        q1, y2 = self.forward(
            augmented1.x,
            augmented1.edge_index,
            augmented1.edge_weight,
            augmented2.x,
            augmented2.edge_index,
            augmented2.edge_weight,
        )
        q2, y1 = self.forward(
            augmented2.x,
            augmented2.edge_index,
            augmented2.edge_weight,
            augmented1.x,
            augmented1.edge_index,
            augmented1.edge_weight,
        )

        # compute cosine similarity loss
        loss = self.criterion(q1, y2, q2, y1)

        # optionally add spatial regularization term to loss
        if hasattr(batch, "position") and self.hparams.spatial_regularization_strength > 0:
            coords = batch.position.to(self.device)
            z = self.net.online_encoder(batch.x, batch.edge_index, batch.edge_weight)

            if batch.x.size(0) > 5000:
                node_subset_sz = self.hparams.node_subset_sz
                cell_random_subset_1 = torch.randint(0, z.shape[0], (node_subset_sz,)).to(
                    self.device
                )
                cell_random_subset_2 = torch.randint(0, z.shape[0], (node_subset_sz,)).to(
                    self.device
                )
                z1, z2 = z[cell_random_subset_1], z[cell_random_subset_2]
                c1, c2 = coords[cell_random_subset_1], coords[cell_random_subset_2]
                pdist = torch.nn.PairwiseDistance(p=2)
                z_dists = pdist(z1, z2) / torch.max(pdist(z1, z2))
                sp_dists = pdist(c1, c2) / torch.max(pdist(c1, c2))
                n_items = z_dists.size(0)
            else:
                z_dists = torch.cdist(z, z, p=2) / torch.max(torch.cdist(z, z, p=2))
                sp_dists = torch.cdist(coords, coords, p=2) / torch.max(
                    torch.cdist(coords, coords, p=2)
                )
                n_items = z.size(0) ** 2

            penalty_1 = torch.sum((1.0 - z_dists) * sp_dists) / n_items
            loss += self.hparams.spatial_regularization_strength * penalty_1

        # log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        # update target network
        self.net.update_target_network(self.hparams.mm)

        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        # run online encoder
        with torch.no_grad():
            node_embeddings = self.net.online_encoder(batch.x, batch.edge_index, batch.edge_weight)

        # load adata object
        sample_name = batch.sample_name[0]
        file_path = os.path.join(self.hparams.processed_dir, sample_name + ".h5ad")
        adata = sc.read_h5ad(file_path)

        # append cell embeddings to adata object
        cell_embeddings_np = node_embeddings.cpu().numpy()
        adata.obsm["cell_embeddings"] = cell_embeddings_np

        # get ground truth labels
        domain_name = None
        if sample_name.startswith("MERFISH_small"):
            domain_name = "domain"
        elif sample_name.startswith("STARmap"):
            domain_name = "region"
        elif sample_name.startswith("BaristaSeq"):
            domain_name = "layer"
        elif sample_name.startswith("Zhuang"):
            domain_name = "parcellation_division_color"
        ground_truth_labels = adata.obs[domain_name]

        # determine resolution based on number of ground truth labels
        sc.pp.neighbors(adata, use_rep="cell_embeddings")
        resolution = set_leiden_resolution(
            adata, target_num_clusters=ground_truth_labels.nunique()
        )
        # perform leiden clustering
        sc.tl.leiden(adata, resolution=resolution)
        leiden_labels = adata.obs["leiden"]

        # convert ground truth labels and leiden labels to PyTorch tensors
        ground_truth_labels = adata.obs[domain_name].astype("category").cat.codes
        ground_truth_labels = torch.tensor(ground_truth_labels.values, dtype=torch.long)
        leiden_labels = adata.obs["leiden"].astype("category").cat.codes
        leiden_labels = torch.tensor(leiden_labels.values, dtype=torch.long)

        # calculate metrics
        nmi = self.test_nmi(ground_truth_labels, leiden_labels)
        ari = self.test_ars(ground_truth_labels, leiden_labels)
        homogeneity = self.test_homogeneity(ground_truth_labels, leiden_labels)
        completeness = self.test_completeness(ground_truth_labels, leiden_labels)

        # log metrics for each graph
        self.log("test/batch_idx", batch_idx, on_step=True, on_epoch=False, prog_bar=False)
        self.log("test/nmi", nmi, on_step=True, on_epoch=False, prog_bar=False)
        self.log("test/ari", ari, on_step=True, on_epoch=False, prog_bar=False)
        self.log("test/homogeneity", homogeneity, on_step=True, on_epoch=False, prog_bar=False)
        self.log("test/completeness", completeness, on_step=True, on_epoch=False, prog_bar=False)

        # save metrics for aggregation
        self.test_outputs.append(
            {
                "sample_name": sample_name,
                "nmi": nmi,
                "ari": ari,
                "homogeneity": homogeneity,
                "completeness": completeness,
            }
        )

        # save the updated adata file to the logs directory
        # save_dir = os.path.join(self.logger.save_dir, "adata_files")
        # os.makedirs(save_dir, exist_ok=True)
        # adata.write_h5ad(os.path.join(save_dir, f"{sample_name}.h5ad"), compression="gzip")

    def on_test_epoch_end(self) -> None:
        """
        Aggregate metrics at the end of the test epoch.
        """
        # get the logger's save directory
        save_dir = self.logger.log_dir if hasattr(self.logger, "log_dir") else self.logger.save_dir
        if save_dir is None:
            raise ValueError("Logger does not have a valid save directory.")

        # save graph-level metrics to a CSV file
        file_path = os.path.join(save_dir, "test_results.csv")
        with open(file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=["sample_name", "nmi", "ari", "homogeneity", "completeness"]
            )
            writer.writeheader()
            writer.writerows(self.test_outputs)

        # extract all NMI and ARI scores
        nmi_scores = torch.stack([x["nmi"] for x in self.test_outputs])
        ari_scores = torch.stack([x["ari"] for x in self.test_outputs])
        homogeneity_scores = torch.stack([x["homogeneity"] for x in self.test_outputs])
        completeness_scores = torch.stack([x["completeness"] for x in self.test_outputs])

        # compute mean scores
        mean_nmi = nmi_scores.mean()
        mean_ari = ari_scores.mean()
        mean_homogeneity = homogeneity_scores.mean()
        mean_completeness = completeness_scores.mean()

        # log the mean scores
        self.log("test/nmi_mean", mean_nmi, on_epoch=True, prog_bar=True)
        self.log("test/ari_mean", mean_ari, on_epoch=True, prog_bar=True)
        self.log("test/homogeneity_mean", mean_homogeneity, on_epoch=True, prog_bar=True)
        self.log("test/completeness_mean", mean_completeness, on_epoch=True, prog_bar=True)

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
    _ = BGRLLitModule(None, None, None, None)
