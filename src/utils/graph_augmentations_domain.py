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

# TODO: cluster-level perturbations like position perturbations only for
#       some clusters identified by a clustering algorithm like kmeans
# TODO: implement low-pass filter for features

from collections import defaultdict
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import coalesce, degree, dropout_edge, remove_self_loops
from torch_scatter import scatter_add
from torch_sparse import SparseTensor


def remove_directed_edges(
    edge_index: torch.Tensor, edge_weight: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Removes all directed (asymmetric) edges from the graph.
    Keeps only edges (i, j) where (j, i) also exists.

    Parameters:
    -----------
    edge_index : torch.Tensor
        Edge index of shape [2, E].
    edge_weight : torch.Tensor, optional
        Edge weights of shape [E], if provided.

    Returns:
    --------
    (edge_index, edge_weight): Tuple[Tensor, Tensor or None]
        Filtered edge_index and edge_weight (if given).
    """
    N = edge_index.max().item() + 1
    src, dst = edge_index[0], edge_index[1]

    edge_ids = src * N + dst
    reverse_ids = dst * N + src

    edge_id_set = set(edge_ids.tolist())
    reverse_id_set = set(reverse_ids.tolist())
    symmetric_ids = edge_id_set & reverse_id_set

    if not symmetric_ids:
        return edge_index.new_empty((2, 0)), None if edge_weight is not None else None

    symmetric_ids = torch.tensor(list(symmetric_ids), device=edge_index.device)
    keep_mask = torch.isin(edge_ids, symmetric_ids)

    filtered_edge_index = edge_index[:, keep_mask]
    filtered_edge_weight = edge_weight[keep_mask] if edge_weight is not None else None

    return filtered_edge_index, filtered_edge_weight


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
        edge_index, edge_weight = data.edge_index, data.edge_weight
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
        ew = edge_weight[edge_mask] if edge_weight is not None else None
        ei, ew = remove_directed_edges(ei, ew)
        ei, ew = coalesce(ei, ew, num_nodes)

        # update edge index and weights
        data.edge_index = ei
        data.edge_weight = ew

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
        edge_weight = data.edge_weight
        num_nodes = data.num_nodes
        device = edge_index.device

        # randomly select edges to rewire and new target nodes
        num_edges = edge_index.size(1)
        rewire_mask = torch.rand(num_edges, device=device) < self.p_rewire
        new_targets = torch.randint(0, num_nodes, (rewire_mask.sum(),), device=device)

        # rewire the selected edges (including reverse edges)
        edge_index = edge_index.clone()
        edge_weight = edge_weight.clone()
        edge_index[1, rewire_mask] = new_targets
        edge_index = torch.cat([edge_index, edge_index[:, rewire_mask].flip(0)], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight[rewire_mask]], dim=0)

        # remove self-loops and directed edges created by rewiring
        edge_index, edge_weight = remove_directed_edges(edge_index, edge_weight)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes=num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # update the new graph
        data.edge_index = edge_index
        data.edge_weight = edge_weight

        assert data.is_undirected(), "Graph is not undirected after rewiring!"
        assert not data.has_self_loops(), "Graph contains self-loops after rewiring!"

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
        num_nodes = data.position.size(0)
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

        # apply the same swaps to the positions
        pos = data.position.clone()
        for a, b in swaps:
            tmp = pos[a].clone()
            pos[a] = pos[b]
            pos[b] = tmp
        data.position = pos

        assert data.is_undirected(), "Graph is not undirected after shuffling!"
        assert not data.has_self_loops(), "Graph has self-loops after shuffling!"

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


class AddEdgesByFeatureSimilarity:
    """
    Adds edges from randomly selected nodes to other feature-similar nodes
    they are not already connected to.

    Parameters:
    -----------
    p_add : float
        Probability of each node being selected for edge addition.
    k : int
        Number of similar nodes to connect to (per selected node).
    similarity : str
        Feature similarity metric: "cosine" or "euclidean".
    """

    def __init__(self, p_add, k_add, similarity="cosine"):
        assert 0.0 < p_add <= 1.0, "Adding probability must be between 0 and 1."
        self.p_add = p_add
        self.k = k_add
        assert similarity in {"cosine", "euclidean"}
        self.similarity = similarity

    def __call__(self, data):
        """
        Applies the edge addition transformation to the input graph data.

        Parameters:
        -----------
        data : Data
            The input graph data containing node positions and edges.

        Returns:
        --------
        Data
            The transformed graph data with added edges.
        """
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        num_nodes = data.num_nodes
        device = data.edge_index.device
        x = data.x

        # select candidate nodes randomly
        selected_mask = torch.rand(num_nodes, device=device) < self.p_add
        selected = selected_mask.nonzero(as_tuple=True)[0]
        if selected.numel() == 0:
            return data

        # build SparseTensor adjacency matrix to exclude already-connected neighbors
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))

        # compute full similarity between selected nodes and all nodes
        # (either cosine similarity or euclidean distance converted to similarity)
        x_sel = x[selected]
        if self.similarity == "cosine":
            x_sel = torch.nn.functional.normalize(x_sel, dim=1)
            x_all = torch.nn.functional.normalize(x, dim=1)
            sim_matrix = torch.matmul(x_sel, x_all.T)
        else:
            x_all_sq = x.pow(2).sum(dim=1, keepdim=True)
            x_sel_sq = x_sel.pow(2).sum(dim=1, keepdim=True)
            sim_matrix = -(x_sel_sq - 2 * x_sel @ x.T + x_all_sq.T)

        # create mask for excluding existing neighbors and self
        row, col, _ = adj.coo()
        neighbor_mask = torch.zeros((selected.numel(), num_nodes), dtype=torch.bool, device=device)
        for idx, i in enumerate(selected.tolist()):
            neighbor_mask[idx, i] = True
            neighbors = adj[i].coo()[1]
            neighbor_mask[idx, neighbors] = True

        sim_matrix[neighbor_mask] = -float("inf")

        # get the top-k similar nodes for each selected node
        topk_sim, topk_indices = torch.topk(sim_matrix, k=min(self.k, num_nodes), dim=1)

        # construct new edges and weights
        src_nodes = selected.unsqueeze(1).expand_as(topk_indices)
        dst_nodes = topk_indices

        edge_pairs = torch.stack([src_nodes.flatten(), dst_nodes.flatten()], dim=0)
        edge_pairs_rev = edge_pairs.flip(0)
        all_new_edges = torch.cat([edge_pairs, edge_pairs_rev], dim=1)

        new_weights = topk_sim.flatten().repeat(2)

        # add the new edges and weights to the graph
        data.edge_index = torch.cat([edge_index, all_new_edges], dim=1)
        data.edge_weight = torch.cat([edge_weight, new_weights], dim=0)

        data.edge_index, data.edge_weight = coalesce(
            data.edge_index, data.edge_weight, num_nodes=num_nodes
        )

        assert data.is_undirected(), "Graph is not undirected after adding edges!"
        assert not data.has_self_loops(), "Graph has self-loops after adding edges!"
        return data

    def __repr__(self):
        """
        Returns a string representation of the transformation.

        Returns:
        --------
        str
            A string describing the transformation and its parameters.
        """
        return f"{self.__class__.__name__}(p_add={self.p_add}, k={self.k}, similarity='{self.similarity}')"


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

    def __init__(self, feature_noise_std):
        self.noise_std = feature_noise_std

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
            noise = torch.randn_like(data.x) * self.noise_std
            data.x += noise
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
        return f"{self.__class__.__name__}(spatial_noise_std={self.noise_std})"


class LowPassFilter:
    def __init__(self, filter_strength):
        self.filter_strength = filter_strength

    def __call__(self, data):
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(filter_strength={self.filter_strength})"

class SmoothFeatures:
    """
    Smooths features of each node as a convex combination of it and its neighbors.

    The smoothing is done in the feature space. The features of each nodes are (1-smooth_strength) * feature_own + sum_n(smooth_strength/n * feature_neighbor_i).
    This smoothing simulates transcript leakage from neighboring cells.

    Parameters:
    -----------
    smooth_strength : float
        The strength of the smoothing. Must be between 0 and 1. 0 means no smoothing, 1 means the feature of the node is the average of its neighbors.

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
    - If the smooth_strength is not between 0 and 1, a ValueError is raised.
    """
    def __init__(self, smooth_strength):
        self.smooth_strength = smooth_strength
        if smooth_strength < 0 or smooth_strength > 1:
            raise ValueError(f"Smooth strength must be between 0 and 1. Got {smooth_strength}.")
        self.smooth_strength = smooth_strength

    def __call__(self, data):
        """
        Applies the feature smoothing transformation to the input graph data.

        Parameters:
        -----------
        data : Data
            The input graph data containing node features.

        Returns:
        --------
        Data
            The transformed graph data with smoothed node features.

        Raises:
        -------
        AttributeError
            If the input graph data does not have an `x` attribute.
        """
        if hasattr(data, "x"):
            # add noise to the features
            x = data.x
            edge_index = data.edge_index
            num_nodes = data.num_nodes
            
            src, dest = edge_index

            neighbor_feature_sum = scatter_add(x[dest], src, dim=0, dim_size=num_nodes)
            degree = scatter_add(torch.ones_like(src, dtype=x.dtype), src, dim=0, dim_size=num_nodes)
            degree = degree.clamp(min=1).unsqueeze(-1)
            neighbor_feature_mean = neighbor_feature_sum / degree
            
            x_smoothed = (1.0 - self.smooth_strength) * x + self.smooth_strength * neighbor_feature_mean

            data.x = x_smoothed
        else:
            raise AttributeError("Data object does not have 'x' attribute.")
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(smooth_strength={self.smooth_strength})"


class Apoptosis:
    """
    Graph apoptosis: randomly drop nodes with probability p and rewire dangling edges
    to *all* surviving neighbors of the dropped node, avoiding self-loops and existing connections.
    """
    def __init__(self, apoptosis_p: float):
        if not 0.0 <= apoptosis_p <= 1.0:
            raise ValueError(f"apoptosis_p must be between 0 and 1, got {apoptosis_p}")
        self.p = apoptosis_p

    def __call__(self, data: Data) -> Data:
        if not hasattr(data, 'edge_index'):
            raise AttributeError("Input data must have 'edge_index'")
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        device = edge_index.device

        # 1) Determine survivors
        survive = torch.rand(num_nodes, device=device) > self.p
        survivors = survive.nonzero(as_tuple=False).view(-1)

        # 2) Build undirected adjacency as CSR for fast neighbor lookup
        row = torch.cat([edge_index[0], edge_index[1]], dim=0)
        col = torch.cat([edge_index[1], edge_index[0]], dim=0)
        adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)).to('csr')

        src, dst = edge_index

        # 3) Keep edges where both endpoints survive
        mask_keep = survive[src] & survive[dst]
        kept = edge_index[:, mask_keep]

        new_edges = [kept]

        # 4) Rewire dangling edges to *all* valid surviving neighbors
        def rewire_all(out_idx, in_idx):
            mask = survive[out_idx] & ~survive[in_idx]
            idx = mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                return None
            u = out_idx[idx]
            v = in_idx[idx]
            rewired = []
            for uu, vv in zip(u.tolist(), v.tolist()):
                nei = adj[vv].storage.col()  # neighbors of dropped node
                cand = nei[survive[nei]]
                forb = adj[uu].storage.col()  # existing neighbors of surviving endpoint
                valid = cand[(cand != uu) & (~torch.isin(cand, forb))]
                if valid.numel() > 0:
                    # connect to all valid neighbors
                    for w in valid.tolist():
                        rewired.append((uu, w))
            if not rewired:
                return None
            return torch.tensor(rewired, dtype=torch.long, device=device).t()

        # src survives, dst dies
        e1 = rewire_all(src, dst)
        if e1 is not None:
            new_edges.append(e1)
        # dst survives, src dies (flip to maintain correct orientation)
        e2 = rewire_all(dst, src)
        if e2 is not None:
            new_edges.append(e2.flip(0))

        # 5) Combine and dedupe
        combined = torch.cat(new_edges, dim=1)
        unique = torch.unique(combined.t(), dim=0).t()

        # 6) Remap node indices to compact range
        new_idx = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        new_idx[survivors] = torch.arange(survivors.size(0), device=device)
        data.edge_index = new_idx[unique]

        # 7) Subset features and update num_nodes
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x[survive]
        data.num_nodes = survivors.size(0)

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(apoptosis_p={self.p})"

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
    feature_noise_std: float,
    p_add: float,
    k_add: int,
    smooth_strength: float,
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
    p_rewire : float
        The probability of rewiring an edge. Must be between 0 and 1.
    p_shuffle : float
        The probability of shuffling a node's position. Must be between 0 and 1.
    spatial_noise_std : float
        The standard deviation of the Gaussian noise to be added to the node positions.
    feature_noise_std : float
        The standard deviation of the Gaussian noise to be added to the node features.
    p_add : float
        Probability of each node being selected for edge addition. Must be between 0 and 1.
    k_add : int
        Number of similar nodes to connect to (per selected node).

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

        # shuffle positions
        if (p_shuffle > 0.0) and ("ShufflePositions" in augmentation_list):
            transforms.append(ShufflePositions(p_shuffle))

        # spatial noise
        if (spatial_noise_std > 0.00) and ("SpatialNoise" in augmentation_list):
            transforms.append(SpatialNoise(spatial_noise_std))

        # feature noise
        if (feature_noise_std > 0.00) and ("FeatureNoise" in augmentation_list):
            transforms.append(FeatureNoise(feature_noise_std))

        # add edges by feature similarity
        if (p_add > 0.0) and ("AddEdgesByFeatureSimilarity" in augmentation_list):
            transforms.append(AddEdgesByFeatureSimilarity(p_add, k_add))
        
        # smooth features
        if (smooth_strength > 0.0) and ("SmoothFeatures" in augmentation_list):
            transforms.append(SmoothFeatures(smooth_strength))

        # apoptosis
        if (apoptosis_p > 0.0) and ("Apoptosis" in augmentation_list):
            transforms.append(Apoptosis(apoptosis_p))

        # return the composed transformation
        return Compose(transforms)

    else:
        raise ValueError(f"Unknown augmentation method: {augmentation_mode}")
