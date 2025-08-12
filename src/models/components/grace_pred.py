"""
GRACE-based Graph Neural Network with Prediction Capabilities.

This module defines a GRACE-based prediction model that extends the GRACE encoder
to support both node-level and graph-level predictions for supervised learning tasks.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GlobalAttention,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    Set2Set,
)

from .grace import Encoder


class GRACE_pred(torch.nn.Module):
    """
    GRACE-based Graph Neural Network with prediction capabilities.

    This class extends the GRACE encoder to support both node-level and graph-level predictions.
    It incorporates flexible configurations for graph pooling, node embedding aggregation, and
    additional subgraph-level features.

    Parameters:
    ----------
    num_layer : int, optional
        The number of GNN layers. Default is 2.
    in_channels : int, optional
        The number of input features. Default is 76.
    emb_dim : int, optional
        The dimensionality of node embeddings. Default is 256.
    num_additional_feat : int, optional
        The number of additional subgraph-level features. Default is 0.
    num_node_tasks : int, optional
        The number of node-level tasks. Default is 0.
    num_graph_tasks : int, optional
        The number of graph-level tasks. Default is 1.
    node_embedding_output : str, optional
        The method for aggregating node embeddings across layers. Options are "last", "concat", "max", or "sum".
        Default is "last".
    drop_ratio : float, optional
        The dropout rate applied after each layer. Default is 0.25.
    graph_pooling : str, optional
        The graph pooling method. Options are "sum", "mean", "max", "attention", or "set2set". Default is "max".
    gnn_type : str, optional
        The type of GNN layer to use. Options are "gin", "gcn", "gat", or "graphsage". Default is "gin".
    activation : callable, optional
        The activation function to use. Default is torch.nn.functional.relu.

    Returns:
    -------
    torch.nn.Module
        A GRACE-based module with prediction capabilities for node-level and graph-level tasks.
    """

    def __init__(
        self,
        num_layer=2,
        in_channels=76,
        emb_dim=256,
        num_node_type=30,
        num_additional_feat=0,
        num_feat=0,
        num_node_tasks=0,
        num_graph_tasks=1,
        node_embedding_output="last",
        drop_ratio=0.25,
        graph_pooling="max",
        gnn_type="gin",
        activation=torch.nn.functional.relu,
    ):
        super().__init__()
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_graph_tasks = num_graph_tasks
        self.num_node_tasks = num_node_tasks
        self.num_layer = num_layer
        self.node_embedding_output = node_embedding_output

        # GRACE encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=emb_dim,
            activation=activation,
            base_model=self._get_base_model(gnn_type),
            num_node_type=num_node_type,
            num_feat=num_feat,
            k=num_layer,
            node_embedding_output=node_embedding_output,
        )

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if node_embedding_output == "concat":
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1)
                )
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if node_embedding_output == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For node and graph predictions
        self.mult = 1
        if graph_pooling[:-1] == "set2set":
            self.mult *= 2
        if node_embedding_output == "concat":
            self.mult *= self.num_layer + 1

        node_embedding_dim = self.mult * self.emb_dim
        
        # Graph-level prediction module
        if self.num_graph_tasks > 0:
            self.graph_pred_module = torch.nn.Sequential(
                torch.nn.Linear(node_embedding_dim + num_additional_feat, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(drop_ratio),
                torch.nn.Linear(node_embedding_dim, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(drop_ratio),
                torch.nn.Linear(node_embedding_dim, self.num_graph_tasks),
            )

        # Node-level prediction module
        if self.num_node_tasks > 0:
            self.node_pred_module = torch.nn.Sequential(
                torch.nn.Linear(node_embedding_dim, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(drop_ratio),
                torch.nn.Linear(node_embedding_dim, self.num_node_tasks),
            )

    def _get_base_model(self, gnn_type):
        """Get the base GNN model based on the specified type (custom layers only)."""
        from .gnn import GCNConv, GATConv, GraphSAGEConv, GINConv
        
        if gnn_type == "gin":
            return GINConv
        elif gnn_type == "gcn":
            return GCNConv
        elif gnn_type == "gat":
            return GATConv
        elif gnn_type == "graphsage":
            return GraphSAGEConv
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def from_pretrained(self, ckpt_file):
        """
        Load a pretrained GRACE model.

        This method loads the state dictionary of a pretrained GRACE model from a lightning checkpoint file.

        Parameters:
        ----------
        ckpt_file : str
            The path to the file containing the pretrained model's state dictionary.
        """
        full_state_dict = torch.load(ckpt_file, weights_only=False)["state_dict"]
        
        # Extract encoder weights from GRACE model
        encoder_state_dict = {}
        for k, v in full_state_dict.items():
            if k.startswith("net.encoder."):
                new_key = k.replace("net.encoder.", "")
                encoder_state_dict[new_key] = v
        
        if encoder_state_dict:
            self.encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"Loaded pretrained GRACE encoder weights from {ckpt_file}")
        else:
            print(f"No encoder weights found in {ckpt_file}")

    def forward(
        self, data, node_feat_mask=None, return_node_embedding=False, return_graph_embedding=False
    ):
        """
        Forward pass of the GRACE prediction model.

        Parameters:
        ----------
        data : torch_geometric.data.Data
            Input graph data.
        node_feat_mask : torch.Tensor, optional
            Mask for node features. Default is None.
        return_node_embedding : bool, optional
            Whether to return node embeddings. Default is False.
        return_graph_embedding : bool, optional
            Whether to return graph embeddings. Default is False.

        Returns:
        -------
        list
            List containing node-level and graph-level predictions.
        """
        batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr

        # Apply node feature mask if provided
        if node_feat_mask is not None:
            x = x * node_feat_mask

        # Determine batch vector (single graph -> zeros)
        batch_vec = batch if batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Get node embeddings from GRACE encoder
        node_embedding = self.encoder(x, edge_index, edge_attr)

        # Append graph-level additional features to node embeddings BEFORE pooling (match gnn.GNN_pred)
        if hasattr(data, 'additional_feat') and data.additional_feat is not None:
            add_feat = data.additional_feat.to(node_embedding.device)
            if add_feat.dim() == 1:
                add_feat = add_feat.view(-1, 1)
            add_feat = add_feat[batch_vec]
            node_embedding = torch.cat([node_embedding, add_feat], dim=1)

        # Apply dropout
        if self.drop_ratio > 0:
            node_embedding = F.dropout(node_embedding, p=self.drop_ratio, training=self.training)

        # Graph pooling
        graph_embedding = self.pool(node_embedding, batch_vec)

        # Initialize output list
        output = []

        # Node-level predictions
        if self.num_node_tasks > 0:
            node_pred = self.node_pred_module(node_embedding)
            output.append(node_pred)

        # Graph-level predictions
        if self.num_graph_tasks > 0:
            graph_pred = self.graph_pred_module(graph_embedding)
            output.append(graph_pred)

        # Return additional embeddings if requested
        if return_node_embedding:
            output.append(node_embedding)
        if return_graph_embedding:
            output.append(graph_embedding)

        return output

    def reset_parameters(self):
        """Reset all parameters of the model."""
        self.encoder.reset_parameters()
        if hasattr(self, 'graph_pred_module'):
            for layer in self.graph_pred_module:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if hasattr(self, 'node_pred_module'):
            for layer in self.node_pred_module:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters() 