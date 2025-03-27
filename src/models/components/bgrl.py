"""
BGRL (Bootstrap Graph Representation Learning) Module.

This module implements the BGRL architecture for self-supervised graph representation learning. 
BGRL is designed to learn meaningful node embeddings by leveraging two networks: an online encoder 
and a target encoder. The online encoder learns representations of the input graph, while the target 
encoder provides stable targets for the online encoder to match. A projector is used to map the 
online encoder's output to a latent space for alignment with the target encoder.

Key Components:
---------------
- Online Encoder: Learns graph representations from the input data.
- Target Encoder: Provides stable targets for the online encoder and is updated using a momentum-based 
  moving average of the online encoder's weights.
- Projector: Maps the online encoder's output to a latent space for alignment with the target encoder.

Features:
---------
- Momentum-based target network updates for stable training.
- Flexible architecture that supports custom encoders and projectors.
- Designed for self-supervised learning tasks on graph-structured data.

Classes:
--------
- BGRL: The main class implementing the BGRL architecture.

Usage:
------
The `BGRL` class can be used as part of a self-supervised learning pipeline for graph-based tasks. 
It requires an encoder and a projector as inputs, which can be customized based on the task.

Example:
--------
>>> import torch
>>> from bgrl import BGRL
>>> from some_module import Encoder, Projector
>>> 
>>> # Initialize encoder and projector
>>> encoder = Encoder(input_dim=128, hidden_dim=64)
>>> projector = Projector(input_dim=64, output_dim=32)
>>> 
>>> # Initialize BGRL model
>>> model = BGRL(encoder=encoder, projector=projector)
>>> 
>>> # Example input graphs
>>> online_x = torch.randn(100, 128)  # 100 nodes with 128 features
>>> target_x = torch.randn(100, 128)
>>> 
>>> # Forward pass
>>> online_q, target_y = model(online_x, target_x)
>>> print(online_q.shape, target_y.shape)  # Output: torch.Size([100, 32]) torch.Size([100, 32])
"""

import copy
import torch
from torch import nn
from typing import Tuple

class BGRL(nn.Module):
    """
    BGRL (Bootstrap Graph Representation Learning) architecture for graph representation learning.

    This model is designed for self-supervised learning on graph data. It consists of an online 
    encoder, a target encoder, and a projector. The online encoder learns representations of the 
    input graph, while the target encoder provides stable targets for the online encoder to match. 
    The projector maps the online encoder's output to a space where it can be compared to the target 
    encoder's output.

    Parameters:
    ----------
    encoder : torch.nn.Module
        Encoder network to be duplicated and used in both online and target networks.
    projector : torch.nn.Module
        Projector network used to map the online encoder's output to a latent space.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        projector: torch.nn.Module
    ) -> None:
        """
        Initialize the BGRL model.

        This method sets up the online encoder, target encoder, and projector. The target encoder 
        is a copy of the online encoder, with its weights reinitialized and gradients disabled.

        Parameters:
        ----------
        encoder : torch.nn.Module
            The encoder network.
        projector : torch.nn.Module
            The projector network.
        """
        super().__init__()

        # define online and target networks
        self.online_encoder = encoder
        self.projector = projector
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights and stop gradient for the target encoder
        self.target_encoder.reset_parameters()
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        """
        Get the trainable parameters of the model.

        This method returns the parameters of the online encoder and projector, which are the 
        components of the model that are updated during training.

        Returns:
        -------
        List[torch.nn.Parameter]
            A list of trainable parameters.
        """
        return list(self.online_encoder.parameters()) + list(self.projector.parameters())

    @torch.no_grad()
    def update_target_network(self, mm: float):
        """
        Update the target network using a momentum-based moving average.

        This method updates the weights of the target encoder by blending its current weights with 
        the weights of the online encoder. The blending is controlled by the momentum parameter.

        Parameters:
        ----------
        mm : float
            Momentum used in the moving average update. Must be between 0.0 and 1.0.

        Raises:
        -------
        AssertionError
            If the momentum value is not in the range [0.0, 1.0].
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(
        self, 
        online_x: torch.Tensor, 
        target_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the BGRL model.

        This method processes two views of the input graph through the online and target networks. 
        The online encoder's output is passed through the projector to generate predictions, while 
        the target encoder's output serves as the target for the predictions.

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
        # forward online network and predict with projector
        online_y = self.online_encoder(online_x)
        online_q = self.projector(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        
        return online_q, target_y
    

if __name__ == "__main__":
    _ = BGRL()