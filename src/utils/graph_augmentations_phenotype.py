from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import coalesce, degree, dropout_edge, remove_self_loops
from torch_sparse import SparseTensor


def remove_directed_edges(
    edge_index: torch.Tensor, edge_attr: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Removes all directed (asymmetric) edges from the graph.
    Keeps only edges (i, j) where (j, i) also exists.

    Parameters:
    -----------
    edge_index : torch.Tensor
        Edge index of shape [2, E].
    edge_attr : torch.Tensor, optional
        Edge weights of shape [E], if provided.

    Returns:
    --------
    (edge_index, edge_attr): Tuple[Tensor, Tensor or None]
        Filtered edge_index and edge_attr (if given).
    """
    N = edge_index.max().item() + 1
    src, dst = edge_index[0], edge_index[1]

    edge_ids = src * N + dst
    reverse_ids = dst * N + src

    edge_id_set = set(edge_ids.tolist())
    reverse_id_set = set(reverse_ids.tolist())
    symmetric_ids = edge_id_set & reverse_id_set

    if not symmetric_ids:
        return edge_index.new_empty((2, 0)), None if edge_attr is not None else None

    symmetric_ids = torch.tensor(list(symmetric_ids), device=edge_index.device)
    keep_mask = torch.isin(edge_ids, symmetric_ids)

    filtered_edge_index = edge_index[:, keep_mask]
    filtered_edge_attr = edge_attr[keep_mask] if edge_attr is not None else None

    return filtered_edge_index, filtered_edge_attr


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

    def __init__(self, p, cell_type_feat=0, unassigned_value=22):
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

        # apply the dropout mask to all features
        data.x[drop_mask] = 0

        # if a specific feature index is provided, handle it separately
        if self.cell_type_feat is not None:
            type_mask = drop_mask[:, self.cell_type_feat]
            data.x[type_mask, self.cell_type_feat] = self.unassigned_value

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
        A threshold value to prevent masking unimportant nodes or edges with too high probabilities. Must be between 0 and 1.

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

    def __init__(self, mu, p_lambda, cell_type_feat=0, unassigned_value=22):
        assert 0.0 < p_lambda <= 1.0, "p_lambda must be between 0 and 1, but got %.2f" % p_lambda
        assert 0.0 < mu <= 1.0, "mu must be between 0 and 1, but got %.2f" % mu
        self.mu = mu
        self.p_lambda = p_lambda
        self.cell_type_feat = cell_type_feat
        self.unassigned_value = unassigned_value

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
        edge_index, edge_attr = data.edge_index, data.edge_attr
        num_nodes = data.num_nodes
        device = edge_index.device

        # compute log-degree node importance
        deg = degree(edge_index[0], num_nodes).float()
        log_deg = torch.log1p(deg)
        node_imp = (log_deg - log_deg.mean()) / (log_deg.max() - log_deg.mean() + 1e-8)
        node_imp = torch.clamp(node_imp, min=0)

        # compute node keep probability and remove unimportant nodes
        node_keep_prob = torch.min(
            (1 - node_imp) * self.mu, torch.full_like(node_imp, self.p_lambda)
        )
        node_mask = torch.rand_like(node_keep_prob) > node_keep_prob
        data.x[~node_mask] = 0

        # set cell_type_feat to unassigned_value
        if self.cell_type_feat is not None:
            dropped_nodes = ~node_mask  # shape [num_nodes]
            data.x[dropped_nodes, self.cell_type_feat] = self.unassigned_value

        # compute edge importance = mean(node_i, node_j)
        edge_imp = (node_imp[edge_index[0]] + node_imp[edge_index[1]]) / 2
        edge_imp = (edge_imp - edge_imp.mean()) / (edge_imp.max() - edge_imp.mean() + 1e-8)
        edge_imp = torch.clamp(edge_imp, min=0)

        # compute edge keep probability
        edge_keep_prob = torch.min(
            (1 - edge_imp) * self.mu, torch.full_like(edge_imp, self.p_lambda)
        )
        edge_mask = torch.rand_like(edge_keep_prob) > edge_keep_prob

        # filter both forward and reverse edges
        ei = edge_index[:, edge_mask]
        ew = edge_attr[edge_mask] if edge_attr is not None else None
        ei, ew = remove_directed_edges(ei, ew)

        # update edge index and weights
        data.edge_index = ei
        data.edge_attr = ew

        assert data.is_undirected(), "Graph is not undirected after dropping edges!"

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


class RewireEdgesAll:
    """
    Rewires edges randomly using all nodes as possible new targets for an edge.

    This transformation randomly rewires edges in the graph with a specified probability `p_rewire`.
    For each edge selected for rewiring, a new target node is chosen randomly from all nodes.

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
        # get attributes from the original graph
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        num_nodes = data.num_nodes
        device = edge_index.device

        # randomly select edges to rewire and new target nodes
        num_edges = edge_index.size(1)
        rewire_mask = torch.rand(num_edges, device=device) < self.p_rewire
        new_targets = torch.randint(0, num_nodes, (rewire_mask.sum(),), device=device)

        # rewire the selected edges (including reverse edges)
        edge_index = edge_index.clone()
        edge_attr = edge_attr.clone()
        edge_index[1, rewire_mask] = new_targets
        edge_index = torch.cat([edge_index, edge_index[:, rewire_mask].flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr[rewire_mask]], dim=0)

        # remove directed edges created by rewiring
        edge_index, edge_attr = remove_directed_edges(edge_index, edge_attr)

        # update the new graph
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        assert data.is_undirected(), "Graph is not undirected after rewiring!"

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


class AddEdgesByCellType:
    """
    Adds edges from randomly selected nodes to other nodes of the same cell type,
    excluding already-connected nodes.

    Parameters:
    -----------
    p_add : float
        Probability of each node being selected for edge addition.
    k : int
        Max number of same-cell-type nodes to connect to (per selected node).
    """

    def __init__(self, p_add, k_add):
        assert 0.0 < p_add <= 1.0, "Adding probability must be between 0 and 1."
        self.p_add = p_add
        self.k = k_add

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        num_nodes = data.num_nodes
        device = edge_index.device

        # cell type is at column 0
        cell_types = data.x[:, 0].long()
        selected_mask = torch.rand(num_nodes, device=device) < self.p_add
        selected = selected_mask.nonzero(as_tuple=True)[0]

        if selected.numel() == 0:
            return data

        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))

        new_edges = []
        new_weights = []

        for i in selected.tolist():
            cell_type = cell_types[i].item()
            same_type_mask = cell_types == cell_type
            same_type_indices = same_type_mask.nonzero(as_tuple=True)[0]

            # exclude i and its neighbors
            neighbors = adj[i].coo()[1]
            exclusion_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
            exclusion_mask[neighbors] = False
            exclusion_mask[i] = False

            candidates = same_type_indices[exclusion_mask[same_type_indices]]
            if candidates.numel() == 0:
                continue

            k_sample = min(self.k, candidates.numel())
            chosen = candidates[torch.randperm(candidates.numel(), device=device)[:k_sample]]

            for j in chosen.tolist():
                new_edges.append([i, j])
                new_edges.append([j, i])
                new_weights.append([0, 1.0])
                new_weights.append([0, 1.0])

        if not new_edges:
            return data

        all_new_edges = torch.tensor(new_edges, device=device).T
        all_new_weights = torch.tensor(new_weights, device=device, dtype=edge_attr.dtype)

        data.edge_index = torch.cat([edge_index, all_new_edges], dim=1)
        data.edge_attr = torch.cat([edge_attr, all_new_weights], dim=0)

        assert data.is_undirected(), "Graph is not undirected after adding edges!"

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(p_add={self.p_add}, k={self.k})"


class ShufflePositions:
    """
    Shuffles nodes within a spatial neighborhood.

    This transformation randomly shuffles nodes with a specified probability `p_shuffle`.
    For each node selected for shuffling, its edges are swapped with the edges of one of its neighbors.

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
        num_nodes = data.x.size(0)
        device = data.edge_index.device
        edge_index = data.edge_index

        # randomly select nodes to shuffle
        shuffle_mask = torch.rand(num_nodes, device=device) < self.p_shuffle
        nodes_to_shuffle = shuffle_mask.nonzero(as_tuple=True)[0]

        if nodes_to_shuffle.numel() == 0:
            return data

        # build neighbor dictionary for quick access
        from collections import defaultdict

        neighbor_dict = defaultdict(list)
        for src, tgt in edge_index.t().tolist():
            neighbor_dict[src].append(tgt)

        # store swaps to apply later
        swaps = []
        for node in nodes_to_shuffle.tolist():
            neighbors = neighbor_dict.get(node, [])
            if neighbors:
                neighbor = neighbors[torch.randint(len(neighbors), (1,)).item()]
                swaps.append((node, neighbor))

        if not swaps:
            return data

        # create mapping from old to new IDs
        swap_map = torch.arange(num_nodes, device=device)
        for a, b in swaps:
            tmp = swap_map[a].item()
            swap_map[a] = swap_map[b]
            swap_map[b] = tmp

        # apply swaps to the edge_index
        data.edge_index = swap_map[data.edge_index]

        assert data.is_undirected(), "Graph is not undirected after shuffling!"

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


class FeatureNoise:
    """
    Adds Gaussian noise to node features in a graph.

    This transformation perturbs the node features by adding Gaussian noise with a specified standard deviation.
    It is useful for simulating variability or introducing randomness in graph data during training.

    Parameters:
    -----------
    feature_noise_std : float
        The standard deviation of the Gaussian noise to be added to the node features.

    Methods:
    --------
    __call__(data):
        Applies the feature noise transformation to the input graph data.
    __repr__():
        Returns a string representation of the transformation.

    Notes:
    ------
    - The transformation assumes that the input graph data contains an `x` attribute representing
      the node features.
    - If the `x` attribute is missing, an AttributeError is raised.
    """

    def __init__(self, feature_noise_std, feature_index=1):
        self.noise_std = feature_noise_std
        self.feature_index = feature_index

    def __call__(self, data):
        """
        Applies the feature noise transformation to the input graph data.

        Parameters:
        -----------
        data : Data
            The input graph data containing node features.

        Returns:
        --------
        Data
            The transformed graph data with perturbed node features.

        Raises:
        -------
        AttributeError
            If the input graph data does not have an `x` attribute.
        """
        if hasattr(data, "x"):
            # add noise to the features
            noise = torch.randn_like(data.x[:, self.feature_index]) * self.noise_std
            data.x[:, self.feature_index] += noise
        else:
            raise AttributeError("Data object does not have 'x' attribute.")

        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return f"{self.__class__.__name__}(feature_noise_std={self.noise_std}, feature_index={self.feature_index})"


def get_graph_augmentation(
    augmentation_mode: str,
    augmentation_list: list[str],
    drop_edge_p: float,
    drop_feat_p: float,
    mu: float,
    p_lambda: float,
    p_rewire: float,
    feature_noise_std: float,
    p_add: float,
    k_add: int,
    p_shuffle: float,
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
    augmentation_list : list[str]
        A list of augmentation methods to apply.
    drop_edge_p : float
        The probability of dropping an edge. Must be between 0 and 1.
    drop_feat_p : float
        The probability of dropping a node feature. Must be between 0 and 1.
    mu : float
        A hyperparameter that controls the overall proportion of masking nodes and edges.
    p_lambda : float
        A threshold value to prevent masking unimportant nodes or edges with too high probabilities. Must be between 0 and 1.
    spatial_noise_std : float
        The standard deviation of the Gaussian noise to be added to the node positions.
    feature_noise_std : float
        The standard deviation of the Gaussian noise to be added to the node features.
    p_add : float
        Probability of each node being selected for edge addition. Must be between 0 and 1.
    k_add : int
        Number of similar nodes to connect to (per selected node).
    p_shuffle : float
        The probability of shuffling a node's position. Must be between 0 and 1.

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

        # drop importance
        if (mu > 0.0) and (p_lambda > 0.0) and ("DropImportance" in augmentation_list):
            transforms.append(DropImportance(mu, p_lambda))

        # rewire edges
        if (p_rewire > 0.0) and ("RewireEdges" in augmentation_list):
            transforms.append(RewireEdgesAll(p_rewire))

        # feature noise
        if (feature_noise_std > 0.00) and ("FeatureNoise" in augmentation_list):
            transforms.append(FeatureNoise(feature_noise_std))

        # add edges by feature similarity
        if (p_add > 0.0) and ("AddEdgesByCellType" in augmentation_list):
            transforms.append(AddEdgesByCellType(p_add, k_add))

        # shuffle positions
        if (p_rewire > 0.0) and ("ShufflePositions" in augmentation_list):
            transforms.append(ShufflePositions(p_shuffle))

        # return the composed transformation
        return Compose(transforms)

    else:
        raise ValueError(f"Unknown augmentation method: {augmentation_mode}")
