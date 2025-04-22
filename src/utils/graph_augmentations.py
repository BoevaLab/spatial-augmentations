"""
Graph Augmentation Utilities for Spatial Omics Data.

This module provides utilities for performing graph augmentations, including dropping edges and node features.
These augmentations are commonly used in graph-based machine learning tasks to improve model robustness and
generalization.

Classes:
--------
- DropFeatures: Drops node features with a specified probability.
- DropEdges: Drops edges with a specified probability.
- Compose: Combines multiple transformations into a single callable.

Functions:
----------
- get_graph_augmentation: Creates a composed graph augmentation pipeline based on the specified method and parameters.
"""

from copy import deepcopy

import numpy as np
import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import degree, dropout_edge


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
        node_importance = (log_deg - avg_log_deg) / (
            max_log_deg - avg_log_deg + 1e-8
        )  # avoid division by zero
        node_importance = torch.clamp(node_importance, min=0)  # ensure non-negative importance

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
        edge_importance = torch.clamp(edge_importance, min=0)  # Ensure non-negative importance

        # define edge sampling probability and create dropout mask
        edge_sampling_prob = torch.min(
            (1 - edge_importance) * self.mu,
            torch.tensor(self.p_lambda, device=edge_importance.device),
        )
        edge_drop_mask = torch.rand_like(edge_sampling_prob) < edge_sampling_prob
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

    Parameters:
    -----------
    p_rewire : float
        The probability of rewiring an edge. Must be between 0 and 1.
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

        for i in range(num_edges):
            # randomly rewire edges with probability p_rewire
            if torch.rand(1) < self.p_rewire:
                # get the source and target nodes of the edge, and the neighbors of the source node
                source = data.edge_index[0, i].item()
                target = data.edge_index[1, i].item()
                neighbors = list(set(data.edge_index[1, data.edge_index[0] == source].tolist()))

                # compute the new targets from the neighbors of neighbors minus the current neighbors
                # and the source and target nodes
                neighbors_of_neighbors = []
                for neighbor in neighbors:
                    second_neighbors = data.edge_index[1, data.edge_index[0] == neighbor].tolist()
                    neighbors_of_neighbors.extend(second_neighbors)
                new_targets = list(
                    set(neighbors_of_neighbors) - set(neighbors) - {source} - {target}
                )

                # randomly select a new target from the available targets
                if len(new_targets) > 0:
                    new_target = torch.tensor(
                        new_targets[torch.randint(0, len(new_targets), (1,)).item()],
                        device=data.edge_index.device,
                    )

                    # update the edge index with the new target
                    data.edge_index[1, i] = new_target

                    # update the corresponding reverse edge
                    reverse_edge_mask = (data.edge_index[0] == target) & (
                        data.edge_index[1] == source
                    )
                    reverse_edge_indices = reverse_edge_mask.nonzero(as_tuple=True)[0]
                    reverse_edge_idx = reverse_edge_indices[0].item()
                    data.edge_index[0, reverse_edge_idx] = new_target

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(p_rewire={self.p_rewire})"


def get_graph_augmentation(augmentation_mode, drop_edge_p, drop_feat_p, mu, p_lambda):
    """
    Creates a composed graph augmentation pipeline based on the specified method and parameters.

    This function constructs a sequence of transformations to apply to graph data. The transformations
    include copying the graph, dropping edges, and dropping node features.

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

        # drop importance
        transforms.append(DropImportance(mu, p_lambda))

        # return the composed transformation
        return Compose(transforms)

    else:
        raise ValueError(f"Unknown augmentation method: {augmentation_mode}")
