import torch


def precompute_neighbors(edge_index, num_nodes):
    """
    Precomputes the neighbors of each node in a graph.

    This function constructs a dictionary where each node is mapped to a set of its neighbors
    based on the provided edge index.

    Parameters:
    -----------
    edge_index : torch.Tensor
        A 2D tensor of shape [2, num_edges] representing the edges of the graph. Each column
        specifies a directed edge from the source node (row 0) to the target node (row 1).
    num_nodes : int
        The total number of nodes in the graph.

    Returns:
    --------
    dict
        A dictionary where the keys are node indices (int) and the values are sets of neighboring
        node indices (int).
    """
    neighbors_dict = {i: set() for i in range(num_nodes)}
    for src, tgt in edge_index.t().tolist():
        neighbors_dict[src].add(tgt)
    return neighbors_dict


def precompute_neighbors_tensor(edge_index, num_nodes, max_num_neighbors):
    """
    Precomputes the neighbors of each node in a dense tensor format.

    Parameters:
    -----------
    edge_index : torch.Tensor
        A 2D tensor of shape [2, num_edges] representing the edges of the graph.
    num_nodes : int
        Total number of nodes in the graph.
    max_num_neighbors : int
        The maximum number of neighbors to store per node.

    Returns:
    --------
    torch.Tensor
        A tensor of shape [num_nodes, max_num_neighbors] containing neighbor indices.
        If a node has fewer neighbors, the remaining slots are filled with -1.
    """
    neighbors = torch.full(
        (num_nodes, max_num_neighbors), -1, dtype=torch.long, device=edge_index.device
    )

    # For each edge, insert the target into the source's neighbor list
    counts = torch.zeros(
        num_nodes, dtype=torch.long, device=edge_index.device
    )  # track how many neighbors we have for each node

    for src, tgt in edge_index.t():
        idx = counts[src]
        if idx < max_num_neighbors:
            neighbors[src, idx] = tgt
            counts[src] += 1

    return neighbors
