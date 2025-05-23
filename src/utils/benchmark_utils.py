import numpy as np
import torch
import torch_geometric
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected


def create_domain_graph(
    graph_name: str,
    num_nodes: int,
    num_features: int,
    num_neighbors: int,
    num_classes: int,
    seed: int = 44,
    noise_scale: float = 0.05,
    position_noise_scale: float = 0.2,
) -> Data:
    """
    Create a synthetic graph for domain identification tasks.

    This function generates a graph where nodes are assigned to groups (domains), each with distinct
    features and positions. Edges are constructed using k-nearest neighbors in position space.

    Parameters:
    -----------
    graph_name : str
        Name of the graph sample.
    num_nodes : int
        Number of nodes in the graph.
    num_features : int
        Number of features per node.
    num_neighbors : int
        Number of neighbors for k-NN graph construction.
    num_classes : int
        Number of groups/domains.
    seed : int, optional
        Random seed for reproducibility. Default is 44.
    noise_scale : float, optional
        Standard deviation of feature noise. Default is 0.05.
    position_noise_scale : float, optional
        Standard deviation of position noise. Default is 0.2.

    Returns:
    --------
    Data
        PyTorch Geometric Data object representing the graph.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    group_features = torch.rand((num_classes, num_features), dtype=torch.float)
    group_assignments = torch.randint(0, num_classes, (num_nodes,))
    x = group_features[group_assignments] + noise_scale * torch.randn((num_nodes, num_features))

    group_positions = torch.rand((num_classes, 2), dtype=torch.float)
    positions = group_positions[group_assignments] + position_noise_scale * torch.randn(
        (num_nodes, 2)
    )

    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm="ball_tree").fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    edge_index = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if neighbor != i:
                edge_index.append([i, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, position=positions)

    graph.edge_index = torch_geometric.utils.to_undirected(graph.edge_index)
    graph.edge_weight = torch.ones(graph.edge_index.size(1), dtype=torch.float)
    graph.sample_name = graph_name

    assert graph.is_undirected(), "Graph is not undirected!"
    assert not graph.has_self_loops(), "Graph has self-loops!"
    return graph


def create_phenotype_graph(
    graph_name: str,
    num_nodes: int,
    num_neighbors: int,
    num_classes: int,
    seed: int = 44,
    position_noise_scale: float = 0.2,
) -> Data:
    """
    Create a synthetic graph for phenotype prediction tasks.

    This function generates a graph where nodes represent cells with a type and size. Edges are
    constructed using k-nearest neighbors in position space, with edge attributes encoding type and distance.

    Parameters:
    -----------
    graph_name : str
        Name of the graph sample.
    num_nodes : int
        Number of nodes in the graph.
    num_neighbors : int
        Number of neighbors for k-NN graph construction.
    num_classes : int
        Number of cell types.
    seed : int, optional
        Random seed for reproducibility. Default is 44.
    position_noise_scale : float, optional
        Standard deviation of position noise. Default is 0.2.

    Returns:
    --------
    Data
        PyTorch Geometric Data object representing the graph.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cell_types = torch.randint(0, num_classes, (num_nodes,))
    sizes = 0.2 + 0.6 * torch.rand(num_nodes)
    x = torch.stack([cell_types.float(), sizes], dim=1)

    group_positions = torch.rand((num_classes, 2))
    position_noise = position_noise_scale * torch.randn((num_nodes, 2))
    positions = group_positions[cell_types] + position_noise

    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm="ball_tree").fit(positions)
    distances, indices = nbrs.kneighbors(positions)

    edge_list = []
    edge_attrs = []

    for i, (dists, neighbors) in enumerate(zip(distances, indices)):
        for dist, j in zip(dists, neighbors):
            if i != j:
                edge_list.append([i, j])
                edge_type = 0 if dist < 0.5 else 1
                edge_attrs.append([edge_type, dist])

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr)
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.sample_name = graph_name

    assert graph.is_undirected(), "Graph is not undirected!"
    return graph
