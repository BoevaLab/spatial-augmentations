"""
Graph Augmentation Utilities for Spatial Omics Data.

This module provides utilities for performing graph augmentations, including dropping edges, dropping node features,
rewiring edges, shuffling node positions, and dropping features/edges based on their importance.
These augmentations are commonly used in graph-based machine learning tasks to improve model robustness,
generalization, and performance on spatial omics data.

Classes:
--------
- DropFeatures: Drops node features with a specified probability.
- DropEdges: Drops edges with a specified probability.
- DropImportance: Drops node features and edges based on their importance.
- RewireEdges: Rewires edges in a spatial neighborhood with a specified probability.
- ShufflePositions: Shuffles node positions within a spatial neighborhood.

Functions:
----------
- get_graph_augmentation: Creates a composed graph augmentation pipeline based on the specified method and parameters.
"""

# TODO: implement spatial noise augmentation
# TODO: implement long range connection augmentation

from copy import deepcopy

import numpy as np
import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import degree, dropout_edge
from torch_sparse import SparseTensor


class DropFeatures:
    """
    Drops node features with a specified probability.

    This transformation randomly masks node features with a probability `p`. If a specific feature index
    (e.g., `cell_type_feat`) is provided, it can apply a custom transformation to that feature, such as
    replacing its value with a default value.

    Parameters:
    -----------
    p : float
        The probability of dropping a feature. Must be between 0 and 1.
    cell_type_feat : int, optional
        The index of the feature to treat as "cell type". If None, no special handling is applied. Default is None.
    unassigned_value : float, optional
        The value to assign to the "cell type" feature when it is dropped. Default is 0.

    Methods:
    --------
    __call__(data):
        Applies the feature dropout transformation to the input graph data.
    __repr__():
        Returns a string representation of the transformation.
    """

    def __init__(self, p, cell_type_feat=None, unassigned_value=0):
        assert 0.0 < p < 1.0, "Dropout probability has to be between 0 and 1, but got %.2f" % p
        self.p = p
        self.cell_type_feat = cell_type_feat
        self.unassigned_value = unassigned_value

    def __call__(self, data):
        """
        Applies the feature dropout transformation to the input graph data.

        Parameters:
        -----------
        data : Data
            The input graph data containing node features.

        Returns:
        --------
        Data
            The transformed graph data with some features dropped.
        """
        # Create a random dropout mask
        drop_mask = (
            torch.empty(data.x.size(), dtype=torch.float32, device=data.x.device).uniform_(0, 1)
            < self.p
        )

        # If a specific feature index is provided, handle it separately
        if self.cell_type_feat is not None:
            type_mask = drop_mask[:, self.cell_type_feat]
            types = data.x[:, self.cell_type_feat]
            types[type_mask] = self.unassigned_value
            data.x[:, self.cell_type_feat] = types

        # Apply the dropout mask to all features
        data.x[drop_mask] = 0
        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return "{}(p={}, cell_type_feat={}, unassigned_value={})".format(
            self.__class__.__name__, self.p, self.cell_type_feat, self.unassigned_value
        )


class DropEdges:
    """
    Drops edges with a specified probability.

    This transformation randomly removes edges from the graph with a probability `p`. Optionally, it can
    enforce the graph to remain undirected after the transformation.

    Parameters:
    -----------
    p : float
        The probability of dropping an edge. Must be between 0 and 1.
    force_undirected : bool, optional
        Whether to enforce the graph to remain undirected after edge dropout. Default is False.

    Methods:
    --------
    __call__(data):
        Applies the edge dropout transformation to the input graph data.
    __repr__():
        Returns a string representation of the transformation.
    """

    def __init__(self, p, force_undirected=False):
        assert 0.0 < p < 1.0, "Dropout probability has to be between 0 and 1, but got %.2f" % p
        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        """
        Applies the edge dropout transformation to the input graph data.

        Parameters:
        -----------
        data : Data
            The input graph data containing edges.

        Returns:
        --------
        Data
            The transformed graph data with some edges dropped.
        """
        edge_index, edge_mask = dropout_edge(
            data.edge_index, p=self.p, force_undirected=self.force_undirected
        )
        data.edge_index = edge_index
        data.edge_weight = data.edge_weight[edge_mask]
        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return "{}(p={}, force_undirected={})".format(
            self.__class__.__name__, self.p, self.force_undirected
        )


class DropImportance:
    """
    Drops node features and edges based on their importance. This class is inspired by
    DOI: 10.1007/s00521-023-09274-6 and by DOI: 10.48550/arXiv.2010.14945.

    Node importance is derived from the logarithm of degree centrality, normalized, and used to define
    the sampling probability for masking node features. Edge importance is calculated as the mean of
    the importance of the two nodes connected by the edge, normalized, and used to define the sampling
    probability for dropping edges.

    Parameters:
    -----------
    mu : float
        A hyperparameter that controls the overall proportion of masking nodes and edges.
    p_lambda : float
        A threshold value to prevent masking nodes or edges with high importance. Must be between 0 and 1.

    Methods:
    --------
    __call__(data):
        Applies the feature and edge dropout transformations based on importance to the input graph data.
    __repr__():
        Returns a string representation of the transformation.

    Notes:
    ------
    - Node importance is calculated using the logarithm of degree centrality.
    - Edge importance is calculated as the mean importance of the two connected nodes.
    - Reverse edges are automatically handled to ensure consistency in undirected graphs.
    """

    def __init__(self, mu, p_lambda):
        assert 0.0 < p_lambda <= 1.0, "p_lambda must be between 0 and 1, but got %.2f" % p_lambda
        assert 0.0 < mu <= 1.0, "mu must be between 0 and 1, but got %.2f" % mu
        self.mu = mu
        self.p_lambda = p_lambda

    def __call__(self, data):
        """
        Applies the feature and edge dropout transformations based on importance to the input graph data.

        Parameters:
        -----------
        data : Data
            The input graph data containing node features and edge information.

        Returns:
        --------
        Data
            The transformed graph data with some features and edges dropped based on importance.
        """
        # calculate degree centrality for each node
        deg = degree(data.edge_index[0], data.num_nodes).float()
        log_deg = torch.log(deg + 1)

        # normalize the log degree centrality
        max_log_deg = log_deg.max()
        avg_log_deg = log_deg.mean()
        node_importance = (log_deg - avg_log_deg) / (max_log_deg - avg_log_deg + 1e-8)
        node_importance = torch.clamp(node_importance, min=0)

        # define node sampling probability and create dropout mask
        node_sampling_prob = torch.min(
            (1 - node_importance) * self.mu,
            torch.tensor(self.p_lambda, device=node_importance.device),
        )
        node_drop_mask = torch.rand_like(data.x[:, 0]) < node_sampling_prob

        # apply the dropout mask to node features
        data.x[node_drop_mask] = 0

        # calculate edge importance as the mean of the importance of the two connected nodes
        edge_importance = (
            node_importance[data.edge_index[0]] + node_importance[data.edge_index[1]]
        ) / 2

        # normalize edge importance
        max_edge_importance = edge_importance.max()
        avg_edge_importance = edge_importance.mean()
        edge_importance = (edge_importance - avg_edge_importance) / (
            max_edge_importance - avg_edge_importance + 1e-8
        )
        edge_importance = torch.clamp(edge_importance, min=0)

        # define edge sampling probability and create dropout mask
        edge_sampling_prob = torch.min(
            (1 - edge_importance) * self.mu,
            torch.tensor(self.p_lambda, device=edge_importance.device),
        )
        edge_drop_mask = torch.rand_like(edge_sampling_prob) < edge_sampling_prob

        # add reverse edges to dropout mask
        for i in range(data.edge_index.size(1)):
            if edge_drop_mask[i]:
                source = data.edge_index[0, i]
                target = data.edge_index[1, i]
                reverse_edge_mask = (data.edge_index[0] == target) & (data.edge_index[1] == source)
                edge_drop_mask[reverse_edge_mask.nonzero(as_tuple=True)[0]] = True

        # apply the dropout mask to edges
        data.edge_index = data.edge_index[:, ~edge_drop_mask]
        data.edge_weight = data.edge_weight[~edge_drop_mask]

        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return f"{self.__class__.__name__}(mu={self.mu}, p_lambda={self.p_lambda})"


class RewireEdges:
    """
    Rewires edges in a spatial neighborhood.

    This transformation randomly rewires edges in the graph with a specified probability `p_rewire`.
    For each edge selected for rewiring, a new target node is chosen from the neighbors of the source
    node's neighbors, excluding the source node, the current target node, and the source node's direct neighbors.

    Parameters:
    -----------
    p_rewire : float
        The probability of rewiring an edge. Must be between 0 and 1.

    Methods:
    --------
    __call__(data):
        Applies the edge rewiring transformation to the input graph data.
    __repr__():
        Returns a string representation of the transformation.

    Notes:
    ------
    - Reverse edges are updated to maintain consistency in undirected graphs.
    - The rewiring process ensures that the graph structure remains valid by avoiding self-loops
      and preserving connectivity within the graph.
    """

    def __init__(self, p_rewire):
        assert 0.0 < p_rewire <= 1.0, "Rewiring probability must be between 0 and 1."
        self.p_rewire = p_rewire

    def __call__(self, data):
        """
        Applies the edge rewiring transformation to the input graph data.

        Parameters:
        -----------
        data : Data
            The input graph data containing precomputed edges and spatial positions.

        Returns:
        --------
        Data
            The transformed graph data with some edges rewired.
        """
        num_edges = data.edge_index.size(1)

        # build a dictionary of neighbors for each node
        neighbors_dict = {i: set() for i in range(data.num_nodes)}
        for src, tgt in data.edge_index.t().tolist():
            neighbors_dict[src].add(tgt)

        # copy the edge index to avoid in-place modification during loop
        new_edges = data.edge_index.clone()

        # loop over each edge and rewire with probability p_rewire
        for i in range(num_edges):
            if torch.rand(1).item() < self.p_rewire:
                # get source and target nodes of the edge
                src = data.edge_index[0, i].item()
                tgt = data.edge_index[1, i].item()

                # get neighbors of the source node
                neighbors = neighbors_dict[src]

                # get the second neighbors of the source node
                second_hop = set()
                for n in neighbors:
                    second_hop |= neighbors_dict[n]

                # candidates for new target nodes are the second neighbors minus the current neighbors
                # and the source and target nodes
                candidate_targets = list(second_hop - neighbors - {src, tgt})

                if candidate_targets:
                    # randomly select a new target node from the candidates
                    new_tgt = torch.tensor(
                        candidate_targets[torch.randint(0, len(candidate_targets), (1,)).item()],
                        device=data.edge_index.device,
                    )

                    # update the forward edge
                    new_edges[1, i] = new_tgt

                    # update the reverse edge if it exists
                    reverse_mask = (data.edge_index[0] == tgt) & (data.edge_index[1] == src)
                    if reverse_mask.any():
                        reverse_idx = reverse_mask.nonzero(as_tuple=True)[0][0]
                        new_edges[0, reverse_idx] = new_tgt

        data.edge_index = new_edges

        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return f"{self.__class__.__name__}(p_rewire={self.p_rewire})"


class RewireEdgesFast:
    """
    Rewires edges in a spatial neighborhood.

    This transformation randomly rewires edges in the graph with a specified probability `p_rewire`.
    For each edge selected for rewiring, a new target node is chosen from the neighbors of the source
    node's neighbors, excluding the source node, the current target node, and the source node's direct neighbors.

    Parameters:
    -----------
    p_rewire : float
        The probability of rewiring an edge. Must be between 0 and 1.

    Methods:
    --------
    __call__(data):
        Applies the edge rewiring transformation to the input graph data.
    __repr__():
        Returns a string representation of the transformation.

    Notes:
    ------
    - Reverse edges are updated to maintain consistency in undirected graphs.
    - The rewiring process ensures that the graph structure remains valid by avoiding self-loops
      and preserving connectivity within the graph.
    """

    def __init__(self, p_rewire):
        assert 0.0 < p_rewire <= 1.0, "Rewiring probability must be between 0 and 1."
        self.p_rewire = p_rewire

    def __call__(self, data):
        """
        Applies the edge rewiring transformation to the input graph data.

        Parameters:
        -----------
        data : Data
            The input graph data containing precomputed edges and spatial positions.

        Returns:
        --------
        Data
            The transformed graph data with some edges rewired.
        """
        edge_index = data.edge_index.clone()
        num_nodes = data.num_nodes
        num_edges = edge_index.size(1)
        device = edge_index.device

        # build adjacency as SparseTensor
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to(
            device
        )

        # sample edges to rewire
        mask = torch.rand(num_edges, device=device) < self.p_rewire
        edge_index_rewire = edge_index[:, mask]
        edge_index_keep = edge_index[:, ~mask]

        # for each edge to rewire: find 2-hop candidates (via matrix multiplication)
        row, col = edge_index_rewire
        second_hop = adj @ adj
        second_hop.set_diag(0)

        # build mask to exclude direct neighbors
        direct = adj.to_dense()
        candidate_mask = (second_hop.to_dense() > 0) & (direct == 0)

        # pick new target nodes for each edge
        new_edges = []
        for i in range(edge_index_rewire.size(1)):
            # get source and target of edge to rewire
            src = edge_index_rewire[0, i].item()
            tgt = edge_index_rewire[1, i].item()
            # print(f"Rewiring edge {src} -> {tgt}")

            # check if the reverse edge still exists
            col_to_remove = ((edge_index_keep[0] == tgt) & (edge_index_keep[1] == src)).nonzero(
                as_tuple=True
            )[0]
            if col_to_remove.numel() == 0:
                continue

            # remove reverse edge from edge_index_keep
            col_to_remove = (
                ((edge_index_keep[0] == tgt) & (edge_index_keep[1] == src))
                .nonzero(as_tuple=True)[0]
                .item()
            )
            edge_index_keep = torch.cat(
                (edge_index_keep[:, :col_to_remove], edge_index_keep[:, col_to_remove + 1 :]),
                dim=1,
            )

            # get candidates for the new target node (remove src and tgt)
            candidates = candidate_mask[src].nonzero(as_tuple=True)[0]
            candidates = candidates[candidates != src]
            candidates = candidates[candidates != tgt]
            # print(f"Candidates for source {src}: {candidates.tolist()}")

            # if there are candidates, pick one randomly
            if len(candidates) > 0:
                new_tgt = candidates[torch.randint(len(candidates), (1,)).item()]
                # print(f"Rewiring edge {src} -> {tgt} to {src} -> {new_tgt.item()}")

                # add new edge and reverse edge
                new_edges.append([src, new_tgt])
                new_edges.append([new_tgt, src])

        # build new edge_index
        if new_edges:
            new_edges_tensor = torch.tensor(new_edges, device=device).T
            edge_index_final = torch.cat([edge_index_keep, new_edges_tensor], dim=1)
        else:
            edge_index_final = edge_index

        data.edge_index = edge_index_final

        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return f"{self.__class__.__name__}(p_rewire={self.p_rewire})"


class ShufflePositions:
    """
    Shuffles node positions within a spatial neighborhood.

    This transformation randomly shuffles the positions of nodes with a specified probability `p_shuffle`.
    For each node selected for shuffling, its position is swapped with the position of one of its neighbors.

    Parameters:
    -----------
    p_shuffle : float
        The probability of shuffling a node's position. Must be between 0 and 1.

    Methods:
    --------
    __call__(data):
        Applies the position shuffling transformation to the input graph data.
    __repr__():
        Returns a string representation of the transformation.

    Notes:
    ------
    - The transformation ensures that the graph structure remains valid by only swapping positions
      between nodes that are connected by an edge.
    - The edge index is updated to reflect the swapped positions, maintaining consistency in the graph.
    """

    def __init__(self, p_shuffle):
        assert 0.0 < p_shuffle <= 1.0, "Shuffling probability must be between 0 and 1."
        self.p_shuffle = p_shuffle

    def __call__(self, data):
        """
        Applies the position shuffling transformation to the input graph data.

        Parameters:
        -----------
        data : Data
            The input graph data containing node positions and edges.

        Returns:
        --------
        Data
            The transformed graph data with shuffled positions.
        """
        for node in range(data.position.size(0)):
            # iterate over all nodes and randomly select a node to shuffle its position
            if torch.rand(1).item() < self.p_shuffle:
                # get the neighbors of the current node
                neighbors = data.edge_index[1, data.edge_index[0] == node]
                if len(neighbors) > 0:
                    # randomly select a neighbor to swap positions with
                    neighbor = neighbors[torch.randint(0, len(neighbors), (1,))].item()

                    # swap the positions of the current node and the selected neighbor
                    data.position[[node, neighbor]] = data.position[[neighbor, node]]

                    # update the edge index to reflect the swapped positions
                    mask_a = data.edge_index == node
                    mask_b = data.edge_index == neighbor
                    data.edge_index[mask_a] = neighbor
                    data.edge_index[mask_b] = node

        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return f"{self.__class__.__name__}(p_shuffle={self.p_shuffle})"


class SpatialNoise:
    """
    Adds Gaussian noise to node positions in a graph.

    This transformation perturbs the positions of nodes by adding Gaussian noise with a specified standard deviation.
    It is useful for simulating spatial variability or introducing randomness in spatial graph data.

    Parameters:
    -----------
    spatial_noise_std : float
        The standard deviation of the Gaussian noise to be added to the node positions.

    Methods:
    --------
    __call__(data):
        Applies the spatial noise transformation to the input graph data.
    __repr__():
        Returns a string representation of the transformation.

    Notes:
    ------
    - The transformation assumes that the input graph data contains a `position` attribute representing
      the spatial positions of nodes.
    - If the `position` attribute is missing, an AttributeError is raised.
    """

    def __init__(self, spatial_noise_std):
        self.noise_std = spatial_noise_std

    def __call__(self, data):
        """
        Applies the spatial noise transformation to the input graph data.

        Parameters:
        -----------
        data : Data
            The input graph data containing node positions.

        Returns:
        --------
        Data
            The transformed graph data with perturbed node positions.

        Raises:
        -------
        AttributeError
            If the input graph data does not have a `position` attribute.
        """
        if hasattr(data, "position"):
            # add noise to the position
            noise = torch.randn_like(data.position) * self.noise_std
            data.position += noise
        else:
            raise AttributeError("Data object does not have 'position' attribute.")

        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return f"{self.__class__.__name__}(spatial_noise_std={self.noise_std})"


def get_graph_augmentation(
    augmentation_mode: str,
    augmentation_list: list[str],
    drop_edge_p: float,
    drop_feat_p: float,
    mu: float,
    p_lambda: float,
    p_rewire: float,
    p_shuffle: float,
    spatial_noise_std: float,
):
    """
    Creates a composed graph augmentation pipeline based on the specified method and parameters.

    This function constructs a sequence of transformations to apply to graph data. The transformations
    include copying the graph, dropping edges, dropping node features, dropping edges and node features
    based on their importance, rewiring edges in a spatial neighborhood, and shuffling node positions
    in a spatial neighborhood.

    Parameters:
    -----------
    augmentation_mode : str
        The augmentation mode to use. Currently, only 'baseline' is supported.
    drop_edge_p : float
        The probability of dropping an edge. Must be between 0 and 1.
    drop_feat_p : float
        The probability of dropping a node feature. Must be between 0 and 1.
    mu : float
        A hyperparameter that controls the overall proportion of masking nodes and edges.
    p_lambda : float
        A threshold value to prevent masking nodes or edges with high importance. Must be between 0 and 1.
    p_rewire : float
        The probability of rewiring an edge. Must be between 0 and 1.
    p_shuffle : float
        The probability of shuffling a node's position. Must be between 0 and 1.
    spatial_noise_std : float
        The standard deviation of the Gaussian noise to be added to the node positions.

    Returns:
    --------
    Compose
        A composed transformation pipeline to apply to graph data.

    Raises:
    -------
    ValueError
        If an unknown augmentation method is specified.
    """
    if augmentation_mode == "baseline":
        transforms = list()

        # make copy of graph
        transforms.append(deepcopy)

        # drop edges
        if drop_edge_p > 0.0:
            transforms.append(DropEdges(drop_edge_p))

        # drop features
        if drop_feat_p > 0.0:
            transforms.append(DropFeatures(drop_feat_p))

        # return the composed transformation
        return Compose(transforms)

    elif augmentation_mode == "advanced":
        transforms = list()

        # make copy of graph
        transforms.append(deepcopy)

        # drop edges
        if drop_edge_p > 0.0:
            transforms.append(DropEdges(drop_edge_p, force_undirected=True))

        # drop features
        if drop_feat_p > 0.0:
            transforms.append(DropFeatures(drop_feat_p))

        # drop importance
        # if mu > 0.0 and p_lambda > 0.0:
        #    transforms.append(DropImportance(mu, p_lambda))

        # rewire edges
        # if p_rewire > 0.0:
        #    transforms.append(RewireEdges(p_rewire))

        # shuffle positions
        # if p_shuffle > 0.0:
        #    transforms.append(ShufflePositions(p_shuffle))

        # spatial noise
        if spatial_noise_std > 0.0:
            transforms.append(SpatialNoise(spatial_noise_std))

        # return the composed transformation
        return Compose(transforms)

    else:
        raise ValueError(f"Unknown augmentation method: {augmentation_mode}")
