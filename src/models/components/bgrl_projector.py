"""
BGRL Projector Module.

This module defines the `BGRLProjector` class, a simple multi-layer perceptron (MLP) used as a projector 
in the Bootstrap Graph Representation Learning (BGRL) framework. The projector maps node embeddings 
from the online encoder into a latent space for alignment with the target encoder in self-supervised 
learning tasks.

The `BGRLProjector` is designed to process node embeddings and project them into a space where they 
can be compared to the output of the target encoder. It consists of two fully connected layers with 
a PReLU activation function in between.

Classes:
--------
- BGRLProjector: A multi-layer perceptron for projecting node embeddings.

Usage:
------
The `BGRLProjector` is typically used as part of the BGRL model to process embeddings from the online 
encoder. It can be initialized with custom input, hidden, and output sizes to match the requirements 
of the task.

Example:
--------
>>> projector = BGRLProjector(input_size=128, output_size=64, hidden_size=32)
>>> x = torch.randn(100, 128)  # 100 nodes with 128-dimensional embeddings
>>> projected_x = projector(x)
>>> print(projected_x.shape)  # Output: torch.Size([100, 64])
"""

import torch
from torch import nn


class BGRLProjector(nn.Module):
    """
    A simple multi-layer perceptron (MLP) used as a projector in the BGRL framework.

    This model consists of two fully connected layers with a PReLU activation function in between.
    It is designed to process node embeddings and project them into a latent space for alignment
    with the target network in self-supervised learning tasks.

    The projector is typically used to map the output of the online encoder to a space where it can
    be compared to the output of the target encoder.

    Attributes:
    ----------
    net : nn.Sequential
        A sequential container of layers, including two linear layers and a PReLU activation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64
    ) -> None:
        """
        Initialize the `BGRLProjector` module.

        Parameters:
        ----------
        input_size : int
            The dimensionality of the input embeddings.
        output_size : int
            The dimensionality of the output embeddings.
        hidden_size : int, optional
            The dimensionality of the hidden layer. Default is 64.
        """
        super().__init__()

        # define projector network
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        
        # initialize weights using Kaiming uniform initialization
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the projector.

        This method applies the sequential layers of the projector to the input tensor.

        Parameters:
        ----------
        x : torch.Tensor
            The input tensor of node embeddings with shape [num_nodes, input_size].

        Returns:
        -------
        torch.Tensor
            The projected embeddings with shape [num_nodes, output_size].
        """
        return self.net(x)
    
    def reset_parameters(self):
        """
        Reset the parameters of the linear layers in the projector.

        This method initializes the weights of all linear layers using Kaiming uniform initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


if __name__ == "__main__":
    _ = BGRLProjector()