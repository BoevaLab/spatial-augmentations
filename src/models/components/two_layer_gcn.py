import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


# TODO: Add batch normalization to the hidden layer
# TODO: Implement weight initialization for GCN layers

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
        dropout: float = 0.5
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
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)
        self.dropout = dropout

    def forward(
            self, 
            x: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_weight: Optional[torch.Tensor] = None
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