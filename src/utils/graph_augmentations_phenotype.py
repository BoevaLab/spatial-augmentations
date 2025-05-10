from copy import deepcopy

import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import dropout_edge


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
        The index of the feature to treat as "cell type". If None, no special handling is applied. Default is 0.
    unassigned_value : float, optional
        The value to assign to the "cell type" feature when it is dropped. Default is 0.

    Methods:
    --------
    __call__(data):
        Applies the feature dropout transformation to the input graph data.
    __repr__():
        Returns a string representation of the transformation.
    """

    def __init__(self, p, cell_type_feat=0, unassigned_value=0):
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
        data.edge_attr = data.edge_attr[edge_mask]
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


def get_graph_augmentation(
    augmentation_mode: str,
    augmentation_list: list[str],
    drop_edge_p: float,
    drop_feat_p: float,
):
    """
    Creates a composed graph augmentation pipeline based on the specified method and parameters.

    Parameters:
    -----------
    augmentation_mode : str
        The augmentation mode to use. Currently, only 'baseline' is supported.
    augmentation_list : list[str]
        A list of augmentation methods to apply.
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
        if (drop_edge_p > 0.0) and ("DropEdges" in augmentation_list):
            transforms.append(DropEdges(drop_edge_p, force_undirected=True))

        # drop features
        if (drop_feat_p > 0.0) and ("DropFeatures" in augmentation_list):
            transforms.append(DropFeatures(drop_feat_p))

        # return the composed transformation
        return Compose(transforms)

    else:
        raise ValueError(f"Unknown augmentation method: {augmentation_mode}")
