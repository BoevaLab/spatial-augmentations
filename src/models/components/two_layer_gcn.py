"""
Two-Layer Graph Convolutional Network (GCN) Module.

This module defines the `TwoLayerGCN` class, a simple two-layer graph convolutional network (GCN)
designed for processing graph-structured data. The model is suitable for tasks such as node
classification, clustering, or domain identification.

The `TwoLayerGCN` consists of:
- Two graph convolutional layers (`GCNConv`) from PyTorch Geometric.
- ReLU activation applied after the first layer.
- Dropout applied between the two layers to prevent overfitting.

Features:
---------
- Supports optional edge weights for weighted graph convolutions.
- Includes a `reset_parameters` method for reinitializing the model's weights.
- Designed to handle graph data in the form of node features and edge indices.

Classes:
--------
- TwoLayerGCN: A two-layer GCN for graph-based learning tasks.

Usage:
------
The `TwoLayerGCN` can be used as a standalone model for graph-based tasks or as a component in
larger graph neural network architectures, e.g., as an encoder in BGRL.

Example:
--------
>>> import torch
>>> from torch_geometric.data import Data
>>> from two_layer_gcn import TwoLayerGCN
>>>
>>> # Example graph data
>>> x = torch.randn(10, 50)  # 10 nodes with 50 features each
>>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 3 edges
>>>
>>> # Initialize the model
>>> model = TwoLayerGCN(input_size=50, hidden_size=32, output_size=16, dropout=0.5)
>>>
>>> # Forward pass
>>> output = model(x, edge_index)
>>> print(output.shape)  # Output: torch.Size([10, 16])
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

# TODO: Add batch normalization to the hidden layer


class TwoLayerGCN(nn.Module):
    """
    A simple two-layer graph convolutional network (GCN) for domain identification.

    This model consists of two graph convolutional layers with ReLU activation and dropout applied
    between the layers. It is designed to process graph-structured data and produce node embeddings
    for downstream tasks such as classification or clustering.
    """

    def __init__(
        self,
        input_size: int = 50,
        hidden_size: int = 32,
        output_size: int = 16,
        dropout: float = 0.5,
    ) -> None:
        """
        Initialize a `TwoLayerGCN` module.

        Parameters:
        ----------
        input_size : int, optional
            The number of input node features. Default is 50.
        hidden_size : int, optional
            The number of output node features of the first convolutional layer. Default is 32.
        output_size : int, optional
            The number of output node features of the final convolutional layer. Default is 16.
        dropout : float, optional
            The dropout rate applied between the two convolutional layers. Default is 0.5.
        """
        super().__init__()

        # define the two graph convolution layers
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)
        self.dropout = dropout

        # initialize the model parameters using Kaiming uniform initialization
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the GCN layers.

        This method reinitializes the weights and biases of the graph convolutional layers (`conv1` and `conv2`)
        to their default initialization values. It is useful for ensuring reproducibility or reinitializing
        the model during training experiments.

        Methods:
        --------
        - `GCNConv.reset_parameters`: Resets the weights and biases of the graph convolutional layers.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform a single forward pass through the two-layer GCN.

        This method applies two graph convolutional layers with ReLU activation and dropout in between.
        Optionally, edge weights can be provided to influence the graph convolution operation.

        Parameters:
        ----------
        x : torch.Tensor
            The input tensor of node features with shape [num_nodes, input_size].
        edge_index : torch.Tensor
            The graph connectivity in COO (coordinate) format with shape [2, num_edges].
        edge_weight : Optional[torch.Tensor], optional
            The edge weights for the graph, with shape [num_edges]. If None, uniform weights are used. Default is None.

        Returns:
        -------
        torch.Tensor
            A tensor of node embeddings with shape [num_nodes, output_size].
        """
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return x


if __name__ == "__main__":
    _ = TwoLayerGCN()
