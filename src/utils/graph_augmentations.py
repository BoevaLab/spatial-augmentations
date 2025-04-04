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

import numpy as np
import torch
from torch_geometric.utils import dropout_edge
from copy import deepcopy
from torch_geometric.transforms import Compose


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
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
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
        drop_mask = torch.empty(data.x.size(), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p

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
        return '{}(p={}, cell_type_feat={}, unassigned_value={})'.format(
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
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
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
        edge_index, edge_mask = dropout_edge(data.edge_index, p=self.p)
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
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)

class PseudoBatchEffect:
    def __init__(self, lambda_param: float, sigma: float):
        """
        Initializes the PseudoBatchEffect transformation.

        Parameters:
        -----------
        lambda_param : float
            The rate parameter for the exponential distribution.
        sigma : float
            The standard deviation for the normal distribution.
        """
        self.lambda_param = lambda_param
        self.sigma = sigma

    def __call__(self, data):
        """
        Applies the pseudo batch effect transformation to the gene expression matrix.

        Parameters:
        -----------
        data : torch_geometric.data.Data
            The graph data object containing the gene expression matrix `x`.

        Returns:
        --------
        data : torch_geometric.data.Data
            The transformed graph data object with updated `x`.
        """
        # gene expression matrix
        x = data.x

        # number of genes
        P = x.shape[1]

        # generate random vectors (following exponential and normal distributions)
        a = torch.from_numpy(np.random.exponential(scale=1 / self.lambda_param, size=P)).float()
        s = torch.from_numpy(np.random.normal(loc=0, scale=self.sigma, size=P)).float()

        # apply the pseudo batch effect transformation
        x_augmented = a + x * (1 + s)

        # update the gene expression matrix in the data object
        data.x = x_augmented

        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return f"{self.__class__.__name__}(lambda_param={self.lambda_param}, sigma={self.sigma})"


class DropImportance:
    """
    Drops node features and edges based on their importance.

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
        assert 0. < p_lambda <= 1., 'p_lambda must be between 0 and 1, but got %.2f' % p_lambda
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
        degree = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float()
        log_degree = torch.log(degree + 1)

        # normalize the log degree centrality
        max_log_degree = log_degree.max()
        avg_log_degree = log_degree.mean()
        node_importance = (log_degree - avg_log_degree) / (max_log_degree - avg_log_degree + 1e-8)  # avoid division by zero
        node_importance = torch.clamp(node_importance, min=0)  # ensure non-negative importance

        # define node sampling probability and create dropout mask
        node_sampling_prob = torch.min((1 - node_importance) * self.mu, torch.tensor(self.p_lambda, device=node_importance.device))
        node_drop_mask = torch.rand_like(data.x[:, 0]) < node_sampling_prob
        data.x[node_drop_mask] = 0

        # calculate edge importance as the mean of the importance of the two connected nodes
        edge_importance = (node_importance[data.edge_index[0]] + node_importance[data.edge_index[1]]) / 2

        # normalize edge importance
        max_edge_importance = edge_importance.max()
        avg_edge_importance = edge_importance.mean()
        edge_importance = (edge_importance - avg_edge_importance) / (max_edge_importance - avg_edge_importance + 1e-8)
        edge_importance = torch.clamp(edge_importance, min=0)  # Ensure non-negative importance

        # dfine edge sampling probability and create dropout mask
        edge_sampling_prob = torch.min((1 - edge_importance) * self.mu, torch.tensor(self.p_lambda, device=edge_importance.device))
        edge_drop_mask = torch.rand_like(edge_sampling_prob) < edge_sampling_prob
        data.edge_index = data.edge_index[:, edge_drop_mask]
        data.edge_weight = data.edge_weight[edge_drop_mask]

        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return '{}(mu={}, p_lambda={})'.format(self.__class__.__name__, self.mu, self.p_lambda)


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
    if augmentation_mode == 'baseline':
        transforms = list()

        # make copy of graph
        transforms.append(deepcopy)

        # drop edges
        if drop_edge_p > 0.:
            transforms.append(DropEdges(drop_edge_p))

        # drop features
        if drop_feat_p > 0.:
            transforms.append(DropFeatures(drop_feat_p))

        # return the composed transformation
        return Compose(transforms)
    
    elif augmentation_mode == 'advanced':
        transforms = list()

        # make copy of graph
        transforms.append(deepcopy)

        # drop importance
        transforms.append(DropImportance(mu, p_lambda))

        # return the composed transformation
        return Compose(transforms)
    
    else:
        raise ValueError('Unknown augmentation method: {}'.format(augmentation_mode))