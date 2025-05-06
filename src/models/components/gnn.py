"""
Graph Neural Network (GNN) Module with Prediction Capabilities.

This module defines several classes for building and using Graph Neural Networks (GNNs) for tasks such as
node-level and graph-level predictions. It includes implementations of popular GNN layers, a general-purpose
GNN module, and an extended GNN module with prediction capabilities.

The module consists of:
- GNN layers (`GINConv`, `GCNConv`, `GATConv`, `GraphSAGEConv`) that implement different message-passing mechanisms.
- A general-purpose GNN class (`GNN`) that supports flexible configurations such as the number of layers,
  embedding dimensions, and node embedding aggregation methods.
- A prediction-capable GNN class (`GNN_pred`) that extends the `GNN` class to support node-level and graph-level
  predictions, with support for various graph pooling methods.

Features:
---------
- Modular design with support for multiple GNN layer types (GIN, GCN, GAT, GraphSAGE).
- Flexible graph pooling methods, including sum, mean, max, attention, and Set2Set pooling.
- Support for node-level and graph-level tasks, with optional additional features.
- Includes a `reset_parameters` method for reinitializing model weights.

Classes:
--------
- GINConv: Graph Isomorphism Network (GIN) convolution layer.
- GCNConv: Graph Convolutional Network (GCN) layer.
- GATConv: Graph Attention Network (GAT) layer.
- GraphSAGEConv: GraphSAGE convolution layer.
- GNN: General-purpose GNN module for generating node embeddings.
- GNN_pred: Extended GNN module with prediction capabilities for node-level and graph-level tasks.

Usage:
------
The classes in this module can be used to build GNN-based models for a variety of graph-based learning tasks,
such as node classification, graph classification, and link prediction.

Example:
--------
>>> import torch
>>> from torch_geometric.data import Data
>>> from gnn import GNN_pred
>>>
>>> # Example graph data
>>> x = torch.randn(10, 38)  # 10 nodes with 38 features each
>>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 3 edges
>>> edge_attr = torch.randn(3, 4)  # 3 edges with 4 attributes each
>>> data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
>>>
>>> # Initialize the model
>>> model = GNN_pred(num_layer=2, num_node_type=20, num_feat=38, emb_dim=256, num_node_tasks=15, num_graph_tasks=2)
>>>
>>> # Forward pass
>>> output = model(data)
>>> print(output)  # Output: List containing node-level and graph-level predictions
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GlobalAttention,
    MessagePassing,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add

NUM_NODE_TYPE = 20  # Default value
NUM_EDGE_TYPE = 4  # neighbor, distant, self


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN) convolution layer with edge information.

    This class implements a GIN layer that incorporates edge information by embedding edge attributes
    and adding them to the node features during message passing.

    Parameters:
    ----------
    emb_dim : int
        Dimensionality of embeddings for nodes and edges.
    aggr : str, optional
        Aggregation method for message passing. Default is "add".

    References:
    ----------
    https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super().__init__()

        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding = torch.nn.Embedding(NUM_EDGE_TYPE, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        """
        Perform a forward pass through the GIN layer.

        This method applies the GIN convolution operation, incorporating edge information
        by embedding edge attributes and adding self-loops to the graph.

        Parameters:
        ----------
        x : torch.Tensor
            Node feature matrix with shape [num_nodes, emb_dim].
        edge_index : torch.Tensor
            Graph connectivity in COO (coordinate) format with shape [2, num_edges].
        edge_attr : torch.Tensor
            Edge attribute matrix with shape [num_edges, edge_attr_dim].

        Returns:
        -------
        torch.Tensor
            Updated node embeddings with shape [num_nodes, emb_dim].
        """
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = NUM_EDGE_TYPE - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        """
        Compute messages to be passed to neighboring nodes.

        This method combines the features of neighboring nodes with the edge attributes.

        Parameters:
        ----------
        x_j : torch.Tensor
            Features of neighboring nodes with shape [num_edges, emb_dim].
        edge_attr : torch.Tensor
            Embedded edge attributes with shape [num_edges, emb_dim].

        Returns:
        -------
        torch.Tensor
            Messages to be aggregated with shape [num_edges, emb_dim].
        """
        return x_j + edge_attr

    def update(self, aggr_out):
        """
        Update node embeddings after aggregation.

        This method applies a multi-layer perceptron (MLP) to the aggregated messages.

        Parameters:
        ----------
        aggr_out : torch.Tensor
            Aggregated messages with shape [num_nodes, emb_dim].

        Returns:
        -------
        torch.Tensor
            Updated node embeddings with shape [num_nodes, emb_dim].
        """
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    """
    Graph Convolutional Network (GCN) layer with edge information.

    This class implements a GCN layer that incorporates edge information by embedding edge attributes
    and normalizing the adjacency matrix during message passing.

    Parameters:
    ----------
    emb_dim : int
        Dimensionality of embeddings for nodes and edges.
    aggr : str, optional
        Aggregation method for message passing. Default is "add".

    References:
    ----------
    https://arxiv.org/abs/1609.02907
    """

    def __init__(self, emb_dim, aggr="add"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Embedding(NUM_EDGE_TYPE, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        # assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        """
        Perform a forward pass through the GCN layer.

        This method applies the GCN convolution operation, incorporating edge information
        by embedding edge attributes, normalizing the adjacency matrix, and adding self-loops.

        Parameters:
        ----------
        x : torch.Tensor
            Node feature matrix with shape [num_nodes, emb_dim].
        edge_index : torch.Tensor
            Graph connectivity in COO (coordinate) format with shape [2, num_edges].
        edge_attr : torch.Tensor
            Edge attribute matrix with shape [num_edges, edge_attr_dim].

        Returns:
        -------
        torch.Tensor
            Updated node embeddings with shape [num_nodes, emb_dim].
        """
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = NUM_EDGE_TYPE - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        """
        Compute messages to be passed to neighboring nodes.

        This method combines the features of neighboring nodes with the edge attributes,
        scaled by the normalization factor.

        Parameters:
        ----------
        x_j : torch.Tensor
            Features of neighboring nodes with shape [num_edges, emb_dim].
        edge_attr : torch.Tensor
            Embedded edge attributes with shape [num_edges, emb_dim].
        norm : torch.Tensor
            Normalization factors for each edge with shape [num_edges].

        Returns:
        -------
        torch.Tensor
            Messages to be aggregated with shape [num_edges, emb_dim].
        """
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    """
    Graph Attention Network (GAT) layer with edge information.

    This class implements a GAT layer that incorporates edge information by embedding edge attributes
    and using attention mechanisms to weigh the importance of neighboring nodes.

    Parameters:
    ----------
    emb_dim : int
        Dimensionality of embeddings for nodes and edges.
    heads : int, optional
        Number of attention heads. Default is 2.
    negative_slope : float, optional
        LeakyReLU angle of the negative slope. Default is 0.2.
    aggr : str, optional
        Aggregation method for message passing. Default is "add".

    References:
    ----------
    https://arxiv.org/abs/1710.10903
    """

    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super().__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding = torch.nn.Embedding(NUM_EDGE_TYPE, heads * emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        """
        Perform a forward pass through the GAT layer.

        This method applies the GAT convolution operation, incorporating edge information
        by embedding edge attributes and computing attention scores for neighboring nodes.

        Parameters:
        ----------
        x : torch.Tensor
            Node feature matrix with shape [num_nodes, emb_dim].
        edge_index : torch.Tensor
            Graph connectivity in COO (coordinate) format with shape [2, num_edges].
        edge_attr : torch.Tensor
            Edge attribute matrix with shape [num_edges, edge_attr_dim].

        Returns:
        -------
        torch.Tensor
            Updated node embeddings with shape [num_nodes, emb_dim].
        """
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = NUM_EDGE_TYPE - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        x = self.weight_linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        """
        Compute messages to be passed to neighboring nodes.

        This method computes attention scores for neighboring nodes and combines their features
        with the edge attributes.

        Parameters:
        ----------
        edge_index : torch.Tensor
            Graph connectivity in COO (coordinate) format with shape [2, num_edges].
        x_i : torch.Tensor
            Features of target nodes with shape [num_edges, heads * emb_dim].
        x_j : torch.Tensor
            Features of source nodes with shape [num_edges, heads * emb_dim].
        edge_attr : torch.Tensor
            Embedded edge attributes with shape [num_edges, heads * emb_dim].

        Returns:
        -------
        torch.Tensor
            Messages to be aggregated with shape [num_edges, heads * emb_dim].
        """
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads * self.emb_dim)

    def update(self, aggr_out):
        """
        Update node embeddings after aggregation.

        This method combines the aggregated messages from multiple attention heads
        and applies a bias term.

        Parameters:
        ----------
        aggr_out : torch.Tensor
            Aggregated messages with shape [num_nodes, heads * emb_dim].

        Returns:
        -------
        torch.Tensor
            Updated node embeddings with shape [num_nodes, emb_dim].
        """
        aggr_out = aggr_out.view(-1, self.heads, self.emb_dim)
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    """
    GraphSAGE convolution layer with edge information.

    This class implements a GraphSAGE layer that incorporates edge information by embedding edge attributes
    and aggregating neighboring node features using mean pooling.

    Parameters:
    ----------
    emb_dim : int
        Dimensionality of embeddings for nodes and edges.
    aggr : str, optional
        Aggregation method for message passing. Default is "mean".

    References:
    ----------
    https://arxiv.org/abs/1706.02216
    """

    def __init__(self, emb_dim, aggr="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Embedding(NUM_EDGE_TYPE, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        """
        Perform a forward pass through the GraphSAGE layer.

        This method applies the GraphSAGE convolution operation, incorporating edge information
        by embedding edge attributes and aggregating neighboring node features.

        Parameters:
        ----------
        x : torch.Tensor
            Node feature matrix with shape [num_nodes, emb_dim].
        edge_index : torch.Tensor
            Graph connectivity in COO (coordinate) format with shape [2, num_edges].
        edge_attr : torch.Tensor
            Edge attribute matrix with shape [num_edges, edge_attr_dim].

        Returns:
        -------
        torch.Tensor
            Updated node embeddings with shape [num_nodes, emb_dim].
        """
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = NUM_EDGE_TYPE - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        """
        Compute messages to be passed to neighboring nodes.

        This method combines the features of neighboring nodes with the edge attributes.

        Parameters:
        ----------
        x_j : torch.Tensor
            Features of neighboring nodes with shape [num_edges, emb_dim].
        edge_attr : torch.Tensor
            Embedded edge attributes with shape [num_edges, emb_dim].

        Returns:
        -------
        torch.Tensor
            Messages to be aggregated with shape [num_edges, emb_dim].
        """
        return x_j + edge_attr

    def update(self, aggr_out):
        """
        Update node embeddings after aggregation.

        This method normalizes the aggregated messages using L2 normalization.

        Parameters:
        ----------
        aggr_out : torch.Tensor
            Aggregated messages with shape [num_nodes, emb_dim].

        Returns:
        -------
        torch.Tensor
            Updated node embeddings with shape [num_nodes, emb_dim].
        """
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """
    General Graph Neural Network (GNN) module.

    This class implements a modular GNN that supports multiple types of graph convolution layers
    (e.g., GIN, GCN, GAT, GraphSAGE) and allows for flexible configurations such as the number of layers,
    embedding dimensions, and node embedding aggregation methods.

    Parameters:
    ----------
    num_layer : int, optional
        The number of GNN layers. Default is 3.
    num_node_type : int, optional
        The number of unique node types. Default is NUM_NODE_TYPE.
    num_feat : int, optional
        The number of additional features besides node type. Default is 38.
    emb_dim : int, optional
        The dimensionality of node and edge embeddings. Default is 256.
    node_embedding_output : str, optional
        The method for aggregating node embeddings across layers. Options are "last", "concat", "max", or "sum".
        Default is "last".
    drop_ratio : float, optional
        The dropout rate applied after each layer. Default is 0.
    gnn_type : str, optional
        The type of GNN layer to use. Options are "gin", "gcn", "gat", or "graphsage". Default is "gin".

    Returns:
    -------
    torch.nn.Module
        A GNN module that outputs node representations.
    """

    def __init__(
        self,
        num_layer=3,
        num_node_type=NUM_NODE_TYPE,
        num_feat=38,
        emb_dim=256,
        node_embedding_output="last",
        drop_ratio=0,
        gnn_type="gin",
    ):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.node_embedding_output = node_embedding_output

        # if self.num_layer < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding = torch.nn.Embedding(num_node_type, emb_dim)
        self.feat_embedding = torch.nn.Linear(num_feat, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.feat_embedding.weight.data)

        self.gnns = torch.nn.ModuleList()
        for _ in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))  # TODO: make modular
            # self.batch_norms.append(GraphNorm(emb_dim))

    def forward(self, *argv):
        """
        Perform a forward pass through the GNN.

        This method applies multiple GNN layers to the input graph, with optional batch normalization
        and dropout between layers. The final node embeddings are aggregated using the specified method.

        Parameters:
        ----------
        *argv : tuple
            The input arguments for the GNN. Supported formats are:
            - (x, edge_index, edge_attr): Node features, edge indices, and edge attributes.
            - (x, edge_index, edge_attr, node_feat_mask): Same as above, with an optional node feature mask.
            - (data): A data object containing x, edge_index, and edge_attr attributes.

        Returns:
        -------
        torch.Tensor
            Node embeddings with shape [num_nodes, emb_dim] or [num_nodes, num_layer * emb_dim]
            depending on the aggregation method.
        """
        if len(argv) == 3:
            node_feat_mask = None
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 4:
            # Support for GNNExplainer
            x, edge_index, edge_attr, node_feat_mask = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            node_feat_mask = None
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("Unmatched number of arguments.")

        # Note that the first feature should always be cell type
        x = self.x_embedding(x[:, 0].long()) + self.feat_embedding(x[:, 1:].float())
        if node_feat_mask is not None:
            x = x * node_feat_mask

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.node_embedding_output == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.node_embedding_output == "last":
            node_representation = h_list[-1]
        elif self.node_embedding_output == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.node_embedding_output == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation

    def reset_parameters(self):
        """
        Reset the parameters of the GNN.

        This method reinitializes the weights of all GNN layers, batch normalization layers,
        and embedding layers to their default values.
        """
        for layer in range(self.num_layer):
            self.gnns[layer].reset_parameters()
            self.batch_norms[layer].reset_parameters()
        self.x_embedding.reset_parameters()
        self.feat_embedding.reset_parameters()


class GNN_pred(torch.nn.Module):
    """
    Graph Neural Network (GNN) with prediction capabilities.

    This class extends the GNN module to support both node-level and graph-level predictions.
    It incorporates flexible configurations for graph pooling, node embedding aggregation, and
    additional subgraph-level features.

    Parameters:
    ----------
    num_layer : int, optional
        The number of GNN layers. Default is 2.
    num_node_type : int, optional
        The number of unique node types. Default is NUM_NODE_TYPE.
    num_feat : int, optional
        The number of additional features besides node type. Default is 38.
    emb_dim : int, optional
        The dimensionality of node and edge embeddings. Default is 256.
    num_additional_feat : int, optional
        The number of additional subgraph-level features. Default is 0.
    num_node_tasks : int, optional
        The number of node-level tasks (e.g., classification or regression). Default is 15.
    num_graph_tasks : int, optional
        The number of graph-level tasks (e.g., classification or regression). Default is 2.
    node_embedding_output : str, optional
        The method for aggregating node embeddings across layers. Options are "last", "concat", "max", or "sum".
        Default is "last".
    drop_ratio : float, optional
        The dropout rate applied after each layer. Default is 0.
    graph_pooling : str, optional
        The graph pooling method. Options are "sum", "mean", "max", "attention", or "set2set". Default is "mean".
    gnn_type : str, optional
        The type of GNN layer to use. Options are "gin", "gcn", "gat", or "graphsage". Default is "gin".

    Returns:
    -------
    torch.nn.Module
        A GNN module with prediction capabilities for node-level and graph-level tasks.

    References:
    ----------
    - GIN: https://arxiv.org/abs/1810.00826
    - JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(
        self,
        num_layer=2,
        num_node_type=NUM_NODE_TYPE,
        num_feat=38,
        emb_dim=256,
        num_additional_feat=0,
        num_node_tasks=15,
        num_graph_tasks=2,
        node_embedding_output="last",
        drop_ratio=0,
        graph_pooling="mean",
        gnn_type="gin",
    ):
        super().__init__()
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_node_tasks = num_node_tasks
        self.num_graph_tasks = num_graph_tasks
        self.num_layer = num_layer

        self.gnn = GNN(
            num_layer,
            num_node_type,
            num_feat,
            emb_dim,
            node_embedding_output=node_embedding_output,
            drop_ratio=drop_ratio,
            gnn_type=gnn_type,
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
        if self.num_graph_tasks > 0:
            self.graph_pred_module = torch.nn.Sequential(
                torch.nn.Linear(node_embedding_dim + num_additional_feat, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(node_embedding_dim, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(node_embedding_dim, self.num_graph_tasks),
            )

        if self.num_node_tasks > 0:
            self.node_pred_module = torch.nn.Sequential(
                torch.nn.Linear(node_embedding_dim + num_additional_feat, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(node_embedding_dim, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(node_embedding_dim, self.num_node_tasks),
            )

    def from_pretrained(self, model_file):
        """
        Load a pretrained GNN model.

        This method loads the state dictionary of a pretrained GNN model from a file.

        Parameters:
        ----------
        model_file : str
            The path to the file containing the pretrained model's state dictionary.
        """
        self.gnn.load_state_dict(torch.load(model_file))  # nosec B614

    def forward(
        self, data, node_feat_mask=None, return_node_embedding=False, return_graph_embedding=False
    ):
        """
        Perform a forward pass through the GNN with prediction capabilities.

        This method applies the GNN to the input graph data and computes predictions for node-level
        and/or graph-level tasks. It supports optional node feature masking and returns embeddings
        if specified.

        Parameters:
        ----------
        data : torch_geometric.data.Data
            A data object containing the graph's node features, edge indices, edge attributes, and
            optionally additional features or batch information.
        node_feat_mask : torch.Tensor, optional
            A mask for node features with shape [num_nodes, emb_dim]. Default is None.
        return_node_embedding : bool, optional
            Whether to return the node embeddings. Default is False.
        return_graph_embedding : bool, optional
            Whether to return the graph embeddings. Default is False.

        Returns:
        -------
        list
            A list containing the following elements, depending on the configuration:
            - Node-level predictions (if `num_node_tasks > 0`).
            - Graph-level predictions (if `num_graph_tasks > 0`).
            - Graph embeddings (if `return_graph_embedding` is True).
            - Node embeddings (if `return_node_embedding` is True).
        """
        gnn_args = [data.x, data.edge_index, data.edge_attr]
        if node_feat_mask is not None:
            assert node_feat_mask.shape[0] == data.x.shape[0]
            gnn_args.append(node_feat_mask.to(data.x.device))
        batch = (
            data.batch if "batch" in data else torch.zeros((len(data.x),)).long().to(data.x.device)
        )

        node_representation = self.gnn(*gnn_args)
        if "additional_feat" in data:
            additional_feat = data.additional_feat[batch]
            node_representation = torch.cat([node_representation, additional_feat], 1)

        return_vals = []
        if self.num_node_tasks > 0:
            if "center_node_index" not in data:
                node_pred = self.node_pred_module(node_representation)
            else:
                center_node_index = (
                    [data.center_node_index]
                    if isinstance(data.center_node_index, int)
                    else data.center_node_index
                )
                center_node_rep = node_representation[center_node_index]
                node_pred = self.node_pred_module(center_node_rep)
            return_vals.append(node_pred)
        if self.num_graph_tasks > 0:
            graph_representation = self.pool(node_representation, batch)
            graph_pred = self.graph_pred_module(graph_representation)
            return_vals.append(graph_pred)
            if return_graph_embedding:
                return_vals.append(graph_representation)
        if return_node_embedding:
            return_vals.append(node_representation)
        return return_vals
