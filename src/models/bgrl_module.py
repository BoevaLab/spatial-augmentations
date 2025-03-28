from typing import Any, Dict, Tuple, List, Callable
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torch.nn.functional import cosine_similarity
from src.utils.graph_augmentations import get_graph_augmentation
from src.utils.schedulers import CosineDecayScheduler


# TODO: add CosineSimilarityScheduler as lr and mm scheduler (including warmup steps and so on)

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
        net : torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        augmentation: str = "baseline",
        mm: int = 0.99,
        drop_edge_p1: float = 0.1,
        drop_edge_p2: float = 0.1,
        drop_feat_p1: float = 0.1,
        drop_feat_p2: float = 0.1
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
        augmentation_method : str, optional
            The graph augmentation method to use. Default is "baseline".
        mm : float, optional
            Momentum for the target network updates. Default is 0.99.
        drop_edge_p1 : float, optional
            Dropout probability for edges in the first view. Default is 0.0.
        drop_edge_p2 : float, optional
            Dropout probability for edges in the second view. Default is 0.0.
        drop_feat_p1 : float, optional
            Dropout probability for features in the first view. Default is 0.0.
        drop_feat_p2 : float, optional
            Dropout probability for features in the second view. Default is 0.0.
        warmup_steps : int, optional
            Number of warmup steps for the learning rate scheduler. Default is 1000.
        num_iterations : int, optional
            Total number of training iterations. Default is 1e5.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # initialize the BGRL model
        self.net = net
        self.augmentation = augmentation

        # loss function
        self.criterion = self.cosine_similarity_loss

        # momentum for the target network updates
        self.mm = mm

        # dropout probabilities for graph augmentation
        self.drop_edge_p1 = drop_edge_p1
        self.drop_edge_p2 = drop_edge_p2
        self.drop_feat_p1 = drop_feat_p1
        self.drop_feat_p2 = drop_feat_p2

        # metrics
        self.train_loss = MeanMetric()

    def forward(
        self, 
        online_x: torch.Tensor,
        online_edge_index: torch.Tensor,
        online_edge_weight: torch.Tensor,
        target_x: torch.Tensor,
        target_edge_index: torch.Tensor,
        target_edge_weight: torch.Tensor
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
        return self.net(online_x, 
                        online_edge_index, 
                        online_edge_weight, 
                        target_x, 
                        target_edge_index, 
                        target_edge_weight
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
        loss = 2 - cosine_similarity(online_q1, target_y2.detach(), dim=-1).mean() \
                 - cosine_similarity(online_q2, target_y1.detach(), dim=-1).mean()
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
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
        transform1 = get_graph_augmentation(self.augmentation, self.drop_edge_p1, self.drop_feat_p1)
        transform2 = get_graph_augmentation(self.augmentation, self.drop_edge_p2, self.drop_feat_p2)

        augmented1 = transform1(batch)
        augmented2 = transform2(batch)
        
        # forward pass
        q1, y2 = self.forward(
            augmented1.x, 
            augmented1.edge_index, 
            augmented1.edge_weight, 
            augmented2.x, 
            augmented2.edge_index, 
            augmented2.edge_weight
        )
        q2, y1 = self.forward(
            augmented2.x, 
            augmented2.edge_index, 
            augmented2.edge_weight, 
            augmented1.x, 
            augmented1.edge_index, 
            augmented1.edge_weight
        )

        # compute loss
        loss = self.criterion(q1, y2, q2, y1)

        # log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        # update target network
        self.net.update_target_network(self.mm)

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
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = BGRLLitModule(None, None, None, None)
