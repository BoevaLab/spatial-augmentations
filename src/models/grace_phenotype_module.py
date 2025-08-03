from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_adj
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from src.models.components.grace import GRACEModel, Encoder, drop_feature
from src.models.components.grace_pred import GRACE_pred
from src.utils import RankedLogger
from src.utils.graph_augmentations_phenotype import get_graph_augmentation
from src.utils.schedulers import WarmupScheduler

log = RankedLogger(__name__, rank_zero_only=True)


class GRACEPhenotypeLitModule(LightningModule):
    """
    A PyTorch Lightning module for training the GRACE (Graph Contrastive Learning) model on phenotype data.

    This module implements the training logic for the GRACE model, which is designed for
    self-supervised learning on graph data using contrastive learning. It supports both 
    pretraining and finetuning modes, with different loss functions and training logic for each mode.

    Key Features:
    - Implements contrastive learning with two augmented views for pretraining
    - Supports binary classification for finetuning
    - Uses graph augmentations for self-supervised learning
    - Logs comprehensive training, validation, and test metrics
    - Configures optimizers and learning rate schedulers

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        mode: str,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        test_thresh: float,
        compile: bool,
        augmentation_mode: str,
        augmentation_list1: List[str],
        augmentation_list2: List[str],
        tau: float,
        drop_edge_p1: float,
        drop_edge_p2: float,
        drop_feat_p1: float,
        drop_feat_p2: float,
        mu: float,
        p_lambda: float,
        p_rewire: float,
        feature_noise_std: float,
        p_add: float,
        k_add: int,
        p_shuffle: float,
        apoptosis_p: float,
        mitosis_p: float,
        shift_p: float,
        shift_map: dict,
        seed: int,
        ckpt_file: str = None,
    ) -> None:
        """
        Initialize the GRACEPhenotypeLitModule.

        Parameters:
        ----------
        mode : str
            The mode of operation. Options are "pretraining", "finetuning", or "evaluation".
        net : torch.nn.Module
            The model to train. For pretraining, this should be a GRACEModel. 
            For finetuning, this should be a GRACE_pred model.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training (e.g., Adam, AdamW).
        scheduler : torch.optim.lr_scheduler
            The learning rate scheduler to use for training (e.g., ReduceLROnPlateau, CosineAnnealingLR).
        test_thresh : float
            The threshold for binary classification during testing.
        compile : bool
            Whether to use Torch's `torch.compile` for model compilation (requires PyTorch 2.0+).
        augmentation_mode : str
            The graph augmentation mode to use. Options are "baseline" or "advanced".
        augmentation_list1 : List[str]
            List of graph augmentation methods to apply for the first view.
        augmentation_list2 : List[str]
            List of graph augmentation methods to apply for the second view.
        tau : float
            Temperature parameter for the contrastive loss.
        drop_edge_p1 : float
            Dropout probability for edges in the first augmented view of the graph.
        drop_edge_p2 : float
            Dropout probability for edges in the second augmented view of the graph.
        drop_feat_p1 : float
            Dropout probability for node features in the first augmented view of the graph.
        drop_feat_p2 : float
            Dropout probability for node features in the second augmented view of the graph.
        mu : float
            Parameter for the graph augmentation methods.
        p_lambda : float
            Parameter for the graph augmentation methods.
        p_rewire : float
            Parameter for the graph augmentation methods.
        feature_noise_std : float
            Standard deviation for the feature noise added during augmentation.
        p_add : float
            Probability of adding new edges during augmentation.
        k_add : int
            Number of new edges to add during augmentation.
        p_shuffle : float
            Probability of shuffling node positions during augmentation.
        apoptosis_p : float
            Probability of apoptosis augmentation.
        mitosis_p : float
            Probability of mitosis augmentation.
        shift_p : float
            Probability of shift augmentation.
        shift_map : dict
            Mapping for shift augmentation.
        seed : int
            Random seed for reproducibility.
        ckpt_file : str, optional
            Path to a pretrained model for finetuning. Default is None.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # initialize the GRACE model
        self.net = net
        self.mode = mode

        # loss metrics (only calculated during training)
        self.train_loss = MeanMetric()

        # validation metrics
        self.val_loss = MeanMetric()
        self.val_auroc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_prec = BinaryPrecision()
        self.val_rec = BinaryRecall()
        self.val_confusion_matrix = BinaryConfusionMatrix()
        self.val_outputs = []

        # test metrics (only calculated during testing)
        self.test_loss = MeanMetric()
        self.test_auroc = BinaryAUROC()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_prec = BinaryPrecision()
        self.test_rec = BinaryRecall()
        self.test_confusion_matrix = BinaryConfusionMatrix()
        self.test_outputs = []

        # load pretrained model if provided
        if ckpt_file is not None:
            self.load_pretrained_model(ckpt_file)

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

    def forward_gnn_pred(self, data: Data) -> list:
        """
        Forward pass for GNN prediction (finetuning mode).

        Parameters:
        ----------
        data : Data
            Input graph data.

        Returns:
        -------
        list
            List containing the predictions.
        """
        # Use the GRACE_pred model for finetuning
        if isinstance(self.net, GRACE_pred):
            return self.net(data)
        else:
            # Fallback for compatibility
            return [self.net(data)]

    def binary_cross_entropy_loss(self, y_pred, y_true, weight, mean=True):
        """
        Compute binary cross-entropy loss.

        Parameters:
        ----------
        y_pred : torch.Tensor
            Predicted logits.
        y_true : torch.Tensor
            Ground truth labels.
        weight : torch.Tensor
            Sample weights.
        mean : bool
            Whether to return mean loss.

        Returns:
        -------
        torch.Tensor
            Binary cross-entropy loss.
        """
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float(), reduction='none')
        if weight is not None:
            loss = loss * weight
        return loss.mean() if mean else loss.sum()

    def freeze_encoder(self):
        """Freeze the encoder parameters (not used in current implementation)."""
        if hasattr(self.net, 'encoder'):
            for param in self.net.encoder.parameters():
                param.requires_grad = False
            log.info("Frozen encoder parameters")

    def unfreeze_encoder(self):
        """Unfreeze the encoder parameters (not used in current implementation)."""
        if hasattr(self.net, 'encoder'):
            for param in self.net.encoder.parameters():
                param.requires_grad = True
            log.info("Unfrozen encoder parameters")

    def load_pretrained_model(self, ckpt_file: str):
        """Load pretrained model weights."""
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load encoder weights
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('net.encoder.'):
                new_key = key.replace('net.encoder.', '')
                encoder_state_dict[new_key] = value
        
        if encoder_state_dict:
            # Load encoder weights directly into the current model if it's a GRACE_pred
            if isinstance(self.net, GRACE_pred):
                self.net.encoder.load_state_dict(encoder_state_dict, strict=False)
                log.info(f"Loaded pretrained encoder weights into GRACE_pred model from {ckpt_file}")
            else:
                # Store encoder weights for use in finetuning
                self.pretrained_encoder_weights = encoder_state_dict
                log.info(f"Stored pretrained encoder weights for later use from {ckpt_file}")
        else:
            log.warning(f"No encoder weights found in {ckpt_file}")

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        This method applies graph augmentations, computes the forward pass, calculates the loss,
        and updates the model parameters.

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
        # Forward pass and loss calculation for pretraining
        if self.mode == "pretraining":
            # Check if we have a GRACEModel for pretraining
            if not hasattr(self.net, 'loss'):
                raise ValueError("For pretraining mode, net must be a GRACEModel with a loss method")
                
            # Get augmented views
            transform1 = get_graph_augmentation(
                self.hparams.augmentation_mode,
                self.hparams.augmentation_list1,
                self.hparams.drop_edge_p1,
                self.hparams.drop_feat_p1,
                self.hparams.mu,
                self.hparams.p_lambda,
                self.hparams.p_rewire,
                self.hparams.feature_noise_std,
                self.hparams.p_add,
                self.hparams.k_add,
                self.hparams.p_shuffle,
                self.hparams.apoptosis_p,
                self.hparams.mitosis_p,
                self.hparams.shift_p,
                self.hparams.shift_map,
            )
            transform2 = get_graph_augmentation(
                self.hparams.augmentation_mode,
                self.hparams.augmentation_list2,
                self.hparams.drop_edge_p2,
                self.hparams.drop_feat_p2,
                self.hparams.mu,
                self.hparams.p_lambda,
                self.hparams.p_rewire,
                self.hparams.feature_noise_std,
                self.hparams.p_add,
                self.hparams.k_add,
                self.hparams.p_shuffle,
                self.hparams.apoptosis_p,
                self.hparams.mitosis_p,
                self.hparams.shift_p,
                self.hparams.shift_map,
            )

            augmented1 = transform1(batch)
            augmented2 = transform2(batch)

            # Forward pass through the model
            z1 = self.net(
                augmented1.x, 
                augmented1.edge_index, 
                edge_weight=augmented1.edge_attr if hasattr(augmented1, 'edge_attr') else None
            )
            z2 = self.net(
                augmented2.x, 
                augmented2.edge_index, 
                edge_weight=augmented2.edge_attr if hasattr(augmented2, 'edge_attr') else None
            )

            # Compute contrastive loss
            loss = self.net.loss(z1, z2, batch_size=0)

        # Forward pass and loss calculation for finetuning
        elif self.mode == "finetuning":
            # Check if we have a prediction model for finetuning
            if not isinstance(self.net, GRACE_pred):
                raise ValueError("For finetuning mode, net must be a GRACE_pred model")
                
            # Forward pass
            res = self.forward_gnn_pred(batch)
            y_pred = res[0].flatten()

            # Compute binary cross-entropy loss
            y_true = batch.y
            weight = batch.w if hasattr(batch, 'w') else None
            loss = self.binary_cross_entropy_loss(y_pred, y_true, weight)

        # No other training mode is supported
        else:
            raise ValueError(
                "Invalid mode for training. Choose either 'pretraining' or 'finetuning'."
            )

        # Log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        return loss

    def calculate_metrics(self, logits, labels, test=False, best_thresh=None) -> dict:
        """
        Evaluate graphs by computing classification metrics.

        Parameters:
        ----------
        logits : torch.Tensor
            The predicted logits for the graphs.
        labels : torch.Tensor
            The ground truth label for the graphs.
        test : bool
            Whether this is for testing (uses fixed threshold).
        best_thresh : float
            The threshold to use for predictions.

        Returns:
        -------
        dict
            A dictionary containing the computed metrics.
        """
        # Turn logits into probabilities and predictions
        probs = torch.sigmoid(logits)

        if not test:
            thresholds = torch.linspace(0.0, 1.0, steps=101).to(probs.device)
            best_f1 = 0.0
            best_thresh = 0.5
            for thresh in thresholds:
                preds = (probs >= thresh).int()
                f1 = self.val_f1(preds, labels)
                if f1 > best_f1:
                    best_f1 = f1.item()
                    best_thresh = thresh.item()

        preds = (probs >= best_thresh).int()

        # Compute metrics
        metrics = {
            "threshold": best_thresh,
            "auroc": self.val_auroc(probs, labels).item(),
            "accuracy": self.val_acc(preds, labels).item(),
            "f1": self.val_f1(preds, labels).item(),
            "precision": self.val_prec(preds, labels).item(),
            "recall": self.val_rec(preds, labels).item(),
        }

        # Compute balanced accuracy
        self.val_confusion_matrix = self.val_confusion_matrix.to("cpu")
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
        # Create a batch of all subgraphs from list of subgraphs
        region_id = batch.region_id[0][0]

        # Run encoder to get predictions for graph
        # Prediction for graph is the mean of all subgraphs
        with torch.no_grad():
                res = self.forward_gnn_pred(batch)
        
        y_preds = res[0].flatten()

        # Get ground truth labels (graph level, same for all subgraphs)
        y_trues = batch.y

        # Calculate the loss for the batch
        weight = batch.w[0] if hasattr(batch, 'w') else None
        loss = self.binary_cross_entropy_loss(y_preds, y_trues, weight)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Save the predictions, ground truth labels, and region_id for the graph
        self.val_outputs.append(
            {
                "y_preds": y_preds.detach().cpu(),  # kept for aggregation with all subgraphs
                "y_true": y_trues[0].detach().cpu(),  # same for all subgraphs of a region
                "region_id": region_id,
            }
        )

    def on_validation_epoch_end(self) -> None:
        """
        Aggregate metrics at the end of the validation epoch and log the results.
        """
        # Aggregate all predictions for each region
        region_preds = defaultdict(list)
        region_trues = {}

        for d in self.val_outputs:
            region_preds[d["region_id"]].append(d["y_preds"])
            region_trues[d["region_id"]] = d["y_true"]

        # Compute metrics for each region
        all_preds = []
        all_trues = []

        for region_id in region_preds.keys():
            # Average predictions across subgraphs for this region
            region_pred = torch.stack(region_preds[region_id]).mean(dim=0)
            region_true = region_trues[region_id]

            all_preds.append(region_pred)
            all_trues.append(region_true)

        # Stack all predictions and truths
        all_preds = torch.stack(all_preds)
        all_trues = torch.stack(all_trues)

        # Calculate metrics
        metrics = self.calculate_metrics(all_preds, all_trues, test=False)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"val/{metric_name}", metric_value, on_epoch=True, prog_bar=True)

        # Clear outputs for next epoch
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
        # Create a batch of all subgraphs from list of subgraphs
        region_id = batch.region_id[0][0]

        # Run encoder to get predictions for graph
        # Prediction for graph is the mean of all subgraphs
        with torch.no_grad():
                res = self.forward_gnn_pred(batch)
        y_preds = res[0].flatten()

        # Get ground truth labels (graph level, same for all subgraphs)
        y_trues = batch.y

        # Calculate the loss for the batch
        weight = batch.w[0] if hasattr(batch, 'w') else None
        loss = self.binary_cross_entropy_loss(y_preds, y_trues, weight)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Save the predictions, ground truth labels, and region_id for the graph
        self.test_outputs.append(
            {
                "y_preds": y_preds.detach().cpu(),  # kept for aggregation with all subgraphs
                "y_true": y_trues[0].detach().cpu(),  # same for all subgraphs of a region
                "region_id": region_id,
            }
        )

    def on_test_epoch_end(self) -> None:
        """
        Aggregate metrics at the end of the test epoch and log the results.
        """
        # Aggregate all predictions for each region
        region_preds = defaultdict(list)
        region_trues = {}

        for d in self.test_outputs:
            region_preds[d["region_id"]].append(d["y_preds"])
            region_trues[d["region_id"]] = d["y_true"]

        # Compute metrics for each region
        all_preds = []
        all_trues = []

        for region_id in region_preds.keys():
            # Average predictions across subgraphs for this region
            region_pred = torch.stack(region_preds[region_id]).mean(dim=0)
            region_true = region_trues[region_id]

            all_preds.append(region_pred)
            all_trues.append(region_true)

        # Stack all predictions and truths
        all_preds = torch.stack(all_preds)
        all_trues = torch.stack(all_trues)

        # Calculate metrics using the test threshold
        metrics = self.calculate_metrics(all_preds, all_trues, test=True, best_thresh=self.hparams.test_thresh)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"test/{metric_name}", metric_value, on_epoch=True, prog_bar=True)

        # Clear outputs for next epoch
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
    # Test instantiation with minimal parameters
    import torch
    from src.models.components.grace import GRACEModel, Encoder
    
    # Create a minimal encoder
    encoder = Encoder(
        in_channels=76,
        out_channels=256,
        activation=torch.nn.functional.relu,
        base_model=torch_geometric.nn.GCNConv,
        k=2
    )
    
    # Create a minimal GRACE model
    net = GRACEModel(
        encoder=encoder,
        num_hidden=256,
        num_proj_hidden=256,
        tau=0.5
    )
    
    # Test instantiation
    _ = GRACEPhenotypeLitModule(
        mode="pretraining",
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        test_thresh=0.5,
        compile=False,
        augmentation_mode="baseline",
        augmentation_list1=[],
        augmentation_list2=[],
        tau=0.5,
        drop_edge_p1=0.3,
        drop_edge_p2=0.4,
        drop_feat_p1=0.3,
        drop_feat_p2=0.4,
        mu=0.0,
        p_lambda=0.0,
        p_rewire=0.0,
        feature_noise_std=0.0,
        p_add=0.0,
        k_add=0,
        p_shuffle=0.0,
        apoptosis_p=0.0,
        mitosis_p=0.0,
        shift_p=0.0,
        shift_map={},
        seed=42,
        warmup_steps=100,
        ckpt_file=None
    ) 