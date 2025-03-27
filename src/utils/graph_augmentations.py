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

import torch
from torch_geometric.utils import dropout_adj
from copy import deepcopy
from torch_geometric.transforms import Compose


class DropFeatures:
    """
    Drops node features with a specified probability.

    This transformation randomly masks node features with a probability `p`. For the `cell_type` feature,
    it replaces the value with an "unassigned" cell type, while other features are set to zero.

    Parameters:
    -----------
    dataset : Dataset
        The dataset containing node features and metadata.
    p : float
        The probability of dropping a feature. Must be between 0 and 1.

    Methods:
    --------
    __call__(data):
        Applies the feature dropout transformation to the input graph data.
    __repr__():
        Returns a string representation of the transformation.
    """

    def __init__(self, dataset, p=None):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p
        self.cell_type_feat = dataset.node_feature_names.index('cell_type')
        self.unassigned_cell_type = dataset.cell_type_mapping[dataset.unassigned_cell_type]

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
        drop_mask = torch.empty(data.x.size(), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        # mask cell type with unassigned cell type, other feats with 0
        type_mask = drop_mask[:, self.cell_type_feat]
        types = data.x[:, self.cell_type_feat]
        types[type_mask] = self.unassigned_cell_type
        data.x[drop_mask] = 0
        data.x[:, self.cell_type_feat] = types
        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return '{}(p={})'.format(self.__class__.__name__, self.p)

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
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)
        # edge_index, edge_id = dropout_edge(edge_index, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
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


def get_graph_augmentation(dataset, augmentation_method, drop_edge_p, drop_feat_p):
    """
    Creates a composed graph augmentation pipeline based on the specified method and parameters.

    This function constructs a sequence of transformations to apply to graph data. The transformations
    include copying the graph, dropping edges, and dropping node features.

    Parameters:
    -----------
    dataset : Dataset
        The dataset containing graph data and metadata.
    augmentation_method : str
        The augmentation method to use. Currently, only 'baseline' is supported.
    drop_edge_p : float
        The probability of dropping an edge. Must be between 0 and 1.
    drop_feat_p : float
        The probability of dropping a node feature. Must be between 0 and 1.

    Returns:
    --------
    Compose
        A composed transformation pipeline to apply to graph data.

    Raises:
    -------
    ValueError
        If an unknown augmentation method is specified.
    """
    if augmentation_method == 'baseline':
        transforms = list()

        # make copy of graph
        transforms.append(deepcopy)

        # drop edges
        if drop_edge_p > 0.:
            transforms.append(DropEdges(drop_edge_p))

        # drop features
        if drop_feat_p > 0.:
            transforms.append(DropFeatures(dataset, drop_feat_p))
        return Compose(transforms)
    
    else:
        raise ValueError('Unknown augmentation method: {}'.format(augmentation_method))