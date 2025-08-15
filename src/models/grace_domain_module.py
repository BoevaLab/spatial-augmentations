import csv
import os
from typing import Any, Dict, List, Tuple

import scanpy as sc
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch_geometric.utils import dropout_adj
from torchmetrics import MeanMetric
from torchmetrics.clustering import (
    AdjustedRandScore,
    CompletenessScore,
    HomogeneityScore,
    NormalizedMutualInfoScore,
)

from src.models.components.grace import GRACEModel, Encoder
from src.utils.clustering_utils import set_leiden_resolution
from src.utils.graph_augmentations_domain import get_graph_augmentation
from src.utils.schedulers import WarmupScheduler


class GRACELitModule(LightningModule):
    """
    A PyTorch Lightning module for training the GRACE (Graph Contrastive Learning) model.

    This module implements the training logic for the GRACE model, which is designed for
    self-supervised learning on graph data using contrastive learning. It creates two
    augmented views of the input graph and learns representations by maximizing the
    mutual information between them.

    Key Features:
    - Implements contrastive learning with two augmented views
    - Supports edge dropout and feature dropout augmentations
    - Uses InfoNCE-style loss with temperature scaling
    - Logs training metrics such as loss
    - Configures optimizers and learning rate schedulers
    - Supports validation and testing with clustering metrics

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: GRACEModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        augmentation_mode: str,
        augmentation_list1: List[str],
        augmentation_list2: List[str],
        tau: float,
        spatial_regularization_strength: float,
        node_subset_sz: int,
        drop_edge_p1: float,
        drop_edge_p2: float,
        drop_feat_p1: float,
        drop_feat_p2: float,
        mu: float,
        p_lambda: float,
        p_rewire: float,
        p_shuffle: float,
        spatial_noise_std: float,
        feature_noise_std: float,
        p_add: float,
        k_add: int,
        smooth_strength: float,
        apoptosis_p: float,
        mitosis_p: float,
        mitosis_feature_noise_std: float,
        processed_dir: str,
        seed: int,
    ) -> None:
        """
        Initialize the GRACELitModule.

        Parameters:
        ----------
        net : GRACEModel
            The GRACE model to train, which includes the encoder and projection head.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training (e.g., Adam, AdamW).
        scheduler : torch.optim.lr_scheduler
            The learning rate scheduler to use for training (e.g., ReduceLROnPlateau, CosineAnnealingLR).
        compile : bool
            Whether to use Torch's `torch.compile` for model compilation (requires PyTorch 2.0+).
        augmentation_mode : str
            The graph augmentation mode to use. Options are "baseline" or "advanced".
        augmentation_list1 : List[str]
            List of graph augmentation methods to apply for the first view. Only necessary if in "advanced" mode.
        augmentation_list2 : List[str]
            List of graph augmentation methods to apply for the second view. Only necessary if in "advanced" mode.
        tau : float
            Temperature parameter for the contrastive loss.
        spatial_regularization_strength : float
            Strength of the spatial regularization term. If set to 0.0, spatial regularization is disabled.
        node_subset_sz : int
            Number of nodes to sample for accelerated spatial regularization when the graph is large.
        drop_edge_p1 : float
            Dropout probability for edges in the first augmented view of the graph.
        drop_edge_p2 : float
            Dropout probability for edges in the second augmented view of the graph.
        drop_feat_p1 : float
            Dropout probability for node features in the first augmented view of the graph.
        drop_feat_p2 : float
            Dropout probability for node features in the second augmented view of the graph.
        mu : float
            A hyperparameter for the graph augmentation process.
        p_lambda : float
            A hyperparameter for the graph augmentation process.
        p_rewire : float
            A hyperparameter for the graph augmentation process.
        p_shuffle : float
            A hyperparameter for the graph augmentation process.
        spatial_noise_std : float
            Standard deviation of the Gaussian noise added to the spatial coordinates of nodes.
        feature_noise_std : float
            Standard deviation of the Gaussian noise added to the node features.
        p_add : float
            Probability of adding a new edges to a node during the graph augmentation process.
        k_add : int
            Number of edges to add for each selected node during the graph augmentation process.
        processed_dir : str
            Directory where processed data is stored. Used during testing to load additional metadata.
        seed : int
            Random seed for reproducibility.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # initialize the GRACE model
        self.net = net

        # loss metrics (only calculated during training)
        self.train_loss = MeanMetric()

        # validation metrics
        self.val_nmi = NormalizedMutualInfoScore()
        self.val_ars = AdjustedRandScore()
        self.val_homogeneity = HomogeneityScore()
        self.val_completeness = CompletenessScore()
        self.val_outputs = []

        # test metrics (only calculated during testing)
        self.test_nmi = NormalizedMutualInfoScore()
        self.test_ars = AdjustedRandScore()
        self.test_homogeneity = HomogeneityScore()
        self.test_completeness = CompletenessScore()
        self.test_outputs = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Perform a forward pass through the GRACE model.

        Parameters:
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices.
        edge_weight : torch.Tensor, optional
            Edge weights.

        Returns:
        -------
        torch.Tensor
            Node embeddings from the encoder.
        """
        return self.net(x, edge_index, edge_weight)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        This method applies graph augmentations, computes the forward pass, and calculates
        the contrastive loss between two augmented views.

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
            self.hparams.augmentation_list1,
            self.hparams.drop_edge_p1,
            self.hparams.drop_feat_p1,
            self.hparams.mu,
            self.hparams.p_lambda,
            self.hparams.p_rewire,
            self.hparams.p_shuffle,
            self.hparams.spatial_noise_std,
            self.hparams.feature_noise_std,
            self.hparams.p_add,
            self.hparams.k_add,
            self.hparams.smooth_strength,
            self.hparams.apoptosis_p,
            self.hparams.mitosis_p,
            self.hparams.mitosis_feature_noise_std,
        )
        transform2 = get_graph_augmentation(
            self.hparams.augmentation_mode,
            self.hparams.augmentation_list2,
            self.hparams.drop_edge_p2,
            self.hparams.drop_feat_p2,
            self.hparams.mu,
            self.hparams.p_lambda,
            self.hparams.p_rewire,
            self.hparams.p_shuffle,
            self.hparams.spatial_noise_std,
            self.hparams.feature_noise_std,
            self.hparams.p_add,
            self.hparams.k_add,
            self.hparams.smooth_strength,
            self.hparams.apoptosis_p,
            self.hparams.mitosis_p,
            self.hparams.mitosis_feature_noise_std,
        )

        augmented1 = transform1(batch)
        augmented2 = transform2(batch)

        # Forward pass through the model
        z1 = self.net(augmented1.x, augmented1.edge_index, augmented1.edge_weight if hasattr(augmented1, 'edge_weight') else None)
        z2 = self.net(augmented2.x, augmented2.edge_index, augmented2.edge_weight if hasattr(augmented2, 'edge_weight') else None)

        # Compute contrastive loss
        loss = self.net.loss(z1, z2, batch_size=z1.size(0))

        # Optionally add spatial regularization term to loss
        if hasattr(batch, "position") and self.hparams.spatial_regularization_strength > 0:
            coords = batch.position.to(self.device)
            z = self.net(batch.x, batch.edge_index, batch.edge_weight if hasattr(batch, 'edge_weight') else None)

            if batch.x.size(0) > 5000:
                node_subset_sz = self.hparams.node_subset_sz
                cell_random_subset_1 = torch.randint(0, z.shape[0], (node_subset_sz,)).to(self.device)
                cell_random_subset_2 = torch.randint(0, z.shape[0], (node_subset_sz,)).to(self.device)
                z1, z2 = z[cell_random_subset_1], z[cell_random_subset_2]
                c1, c2 = coords[cell_random_subset_1], coords[cell_random_subset_2]
                pdist = torch.nn.PairwiseDistance(p=2)
                z_dists = pdist(z1, z2) / torch.max(pdist(z1, z2))
                sp_dists = pdist(c1, c2) / torch.max(pdist(c1, c2))
                n_items = z_dists.size(0)
            else:
                z_dists = torch.cdist(z, z, p=2) / torch.max(torch.cdist(z, z, p=2))
                sp_dists = torch.cdist(coords, coords, p=2) / torch.max(torch.cdist(coords, coords, p=2))
                n_items = z.size(0) ** 2

            penalty_1 = torch.sum((1.0 - z_dists) * sp_dists) / n_items
            loss += self.hparams.spatial_regularization_strength * penalty_1

        # Log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Perform a single test step.

        This method evaluates the model on a batch of test data by:
        - Generating node embeddings using the encoder.
        - Loading the corresponding AnnData object for the batch.
        - Appending the generated embeddings to the AnnData object.
        - Performing Leiden clustering on the embeddings.
        - Comparing the clustering results with ground truth labels using various metrics.

        Parameters:
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A batch of input data containing node features, edge indices, edge weights, and metadata.
        batch_idx : int
            The index of the current batch.

        Returns:
        -------
        None
            Logs the calculated metrics for the batch, including:
            - Normalized Mutual Information (NMI)
            - Adjusted Rand Index (ARI)
            - Homogeneity Score
            - Completeness Score
        """
        # Process each sample in the batch
        graphs = batch.to_data_list()
        for graph in graphs:
            # Run encoder
            with torch.no_grad():
                node_embeddings = self.net(
                    graph.x, 
                    graph.edge_index, 
                    graph.edge_weight if hasattr(graph, 'edge_weight') else None
                )

            # Load the corresponding AnnData object
            sample_name = graph.sample_name
            file_path = os.path.join(self.hparams.processed_dir, sample_name + ".h5ad")
            adata = sc.read_h5ad(file_path)

            # Append cell embeddings to adata object
            cell_embeddings_np = node_embeddings.cpu().numpy()
            adata.obsm["cell_embeddings"] = cell_embeddings_np

            # Get ground truth labels
            ground_truth_labels = adata.obs["domain_annotation"]

            # Determine resolution based on number of ground truth labels
            sc.pp.neighbors(adata, use_rep="cell_embeddings")
            resolution = set_leiden_resolution(
                adata, target_num_clusters=ground_truth_labels.nunique(), seed=self.hparams.seed
            )
            
            # Perform leiden clustering
            sc.tl.leiden(
                adata,
                resolution=resolution,
                flavor="igraph",
                n_iterations=2,
                directed=False,
                random_state=self.hparams.seed,
            )
            leiden_labels = adata.obs["leiden"]

            # Convert ground truth labels and leiden labels to PyTorch tensors
            ground_truth_labels = ground_truth_labels.astype("category").cat.codes
            ground_truth_labels = torch.tensor(ground_truth_labels.values, dtype=torch.long)
            leiden_labels = adata.obs["leiden"].astype("category").cat.codes
            leiden_labels = torch.tensor(leiden_labels.values, dtype=torch.long)

            # Calculate metrics
            nmi = self.test_nmi(ground_truth_labels, leiden_labels)
            ari = self.test_ars(ground_truth_labels, leiden_labels)
            homogeneity = self.test_homogeneity(ground_truth_labels, leiden_labels)
            completeness = self.test_completeness(ground_truth_labels, leiden_labels)

            # Save metrics for aggregation in on_test_epoch_end()
            self.test_outputs.append(
                {
                    "sample_name": sample_name,
                    "nmi": nmi,
                    "ari": ari,
                    "homogeneity": homogeneity,
                    "completeness": completeness,
                }
            )

    def on_test_epoch_end(self) -> None:
        """
        Aggregate metrics at the end of the test epoch.
        """
        # Get the logger's save directory
        save_dir = self.logger.log_dir if hasattr(self.logger, "log_dir") else self.logger.save_dir
        if save_dir is None:
            raise ValueError("Logger does not have a valid save directory.")

        # Save graph-level metrics to a CSV file
        file_path = os.path.join(save_dir, "test_results.csv")
        with open(file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=["sample_name", "nmi", "ari", "homogeneity", "completeness"]
            )
            writer.writeheader()
            writer.writerows(self.test_outputs)

        # Extract all NMI and ARI scores
        nmi_scores = torch.stack([x["nmi"] for x in self.test_outputs])
        ari_scores = torch.stack([x["ari"] for x in self.test_outputs])
        homogeneity_scores = torch.stack([x["homogeneity"] for x in self.test_outputs])
        completeness_scores = torch.stack([x["completeness"] for x in self.test_outputs])

        # Compute mean scores
        mean_nmi = nmi_scores.mean()
        mean_ari = ari_scores.mean()
        mean_homogeneity = homogeneity_scores.mean()
        mean_completeness = completeness_scores.mean()

        # Log the mean scores
        self.log("test/nmi_mean", mean_nmi, on_epoch=True, prog_bar=True)
        self.log("test/ari_mean", mean_ari, on_epoch=True, prog_bar=True)
        self.log("test/homogeneity_mean", mean_homogeneity, on_epoch=True, prog_bar=True)
        self.log("test/completeness_mean", mean_completeness, on_epoch=True, prog_bar=True)

        # Clear the outputs for the next test run
        self.test_outputs.clear()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Perform a single validation step.

        This method evaluates the model on a batch of validation data by:
        - Generating node embeddings using the encoder.
        - Loading the corresponding AnnData object for the batch.
        - Appending the generated embeddings to the AnnData object.
        - Performing Leiden clustering on the embeddings.
        - Comparing the clustering results with ground truth labels using various metrics.

        Parameters:
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A batch of input data containing node features, edge indices, edge weights, and metadata.
        batch_idx : int
            The index of the current batch.

        Returns:
        -------
        None
            Logs the calculated metrics for the batch, including:
            - Normalized Mutual Information (NMI)
            - Adjusted Rand Index (ARI)
            - Homogeneity Score
            - Completeness Score
        """
        # Process each sample in the batch
        graphs = batch.to_data_list()
        for graph in graphs:
            # Run encoder
            with torch.no_grad():
                node_embeddings = self.net(
                    graph.x, 
                    graph.edge_index, 
                    graph.edge_weight if hasattr(graph, 'edge_weight') else None
                )

            # Load the corresponding AnnData object
            sample_name = graph.sample_name
            file_path = os.path.join(self.hparams.processed_dir, sample_name + ".h5ad")
            adata = sc.read_h5ad(file_path)

            # Append cell embeddings to adata object
            cell_embeddings_np = node_embeddings.cpu().numpy()
            adata.obsm["cell_embeddings"] = cell_embeddings_np

            # Get ground truth labels
            ground_truth_labels = adata.obs["domain_annotation"]

            # Determine resolution based on number of ground truth labels
            sc.pp.neighbors(adata, use_rep="cell_embeddings")
            resolution = set_leiden_resolution(
                adata, target_num_clusters=ground_truth_labels.nunique(), seed=self.hparams.seed
            )
            
            # Perform leiden clustering
            sc.tl.leiden(
                adata,
                resolution=resolution,
                flavor="igraph",
                n_iterations=2,
                directed=False,
                random_state=self.hparams.seed,
            )
            leiden_labels = adata.obs["leiden"]

            # Convert ground truth labels and leiden labels to PyTorch tensors
            ground_truth_labels = ground_truth_labels.astype("category").cat.codes
            ground_truth_labels = torch.tensor(ground_truth_labels.values, dtype=torch.long)
            leiden_labels = adata.obs["leiden"].astype("category").cat.codes
            leiden_labels = torch.tensor(leiden_labels.values, dtype=torch.long)

            # Calculate metrics
            nmi = self.val_nmi(ground_truth_labels, leiden_labels)
            ari = self.val_ars(ground_truth_labels, leiden_labels)
            homogeneity = self.val_homogeneity(ground_truth_labels, leiden_labels)
            completeness = self.val_completeness(ground_truth_labels, leiden_labels)

            # Save metrics for aggregation in on_validation_epoch_end()
            self.val_outputs.append(
                {
                    "sample_name": sample_name,
                    "nmi": nmi,
                    "ari": ari,
                    "homogeneity": homogeneity,
                    "completeness": completeness,
                }
            )

    def on_validation_epoch_end(self) -> None:
        """
        Aggregate metrics at the end of the validation epoch.
        """
        # Get the logger's save directory
        save_dir = self.logger.log_dir if hasattr(self.logger, "log_dir") else self.logger.save_dir
        if save_dir is None:
            raise ValueError("Logger does not have a valid save directory.")

        # Save graph-level metrics to a CSV file
        file_path = os.path.join(save_dir, "val_results.csv")
        with open(file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=["sample_name", "nmi", "ari", "homogeneity", "completeness"]
            )
            writer.writeheader()
            writer.writerows(self.val_outputs)

        # Extract all NMI and ARI scores
        nmi_scores = torch.stack([x["nmi"] for x in self.val_outputs])
        ari_scores = torch.stack([x["ari"] for x in self.val_outputs])
        homogeneity_scores = torch.stack([x["homogeneity"] for x in self.val_outputs])
        completeness_scores = torch.stack([x["completeness"] for x in self.val_outputs])

        # Compute mean scores
        mean_nmi = nmi_scores.mean()
        mean_ari = ari_scores.mean()
        mean_homogeneity = homogeneity_scores.mean()
        mean_completeness = completeness_scores.mean()

        # Log the mean scores
        self.log("val/nmi_mean", mean_nmi, on_epoch=True, prog_bar=True)
        self.log("val/ari_mean", mean_ari, on_epoch=True, prog_bar=True)
        self.log("val/homogeneity_mean", mean_homogeneity, on_epoch=True, prog_bar=True)
        self.log("val/completeness_mean", mean_completeness, on_epoch=True, prog_bar=True)

        # Clear the outputs for the next validation run
        self.val_outputs.clear()

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
    _ = GRACELitModule(None, None, None, None) 