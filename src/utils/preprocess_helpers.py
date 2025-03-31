import os
import scanpy as sc
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from scipy.spatial import distance_matrix, cKDTree
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def read_samples_into_dict(file_paths):
    """
    Reads multiple AnnData objects (samples) from a list of file paths and stores them in a dictionary.

    Parameters:
    ----------
    file_paths : list of str
        A list of file paths to the `.h5ad` files to be read.

    Returns:
    -------
    dict
        A dictionary where the keys are the filenames (without extensions) 
        and the values are the corresponding AnnData objects.
    """
    adata_dict = {}
    
    for file in file_paths:
        adata = sc.read_h5ad(file)
        file_name = os.path.splitext(os.path.basename(file))[0]
        adata_dict[file_name] = adata
        
    return adata_dict

def save_sample(adata, graph, output_dir, sample_name):
    """
    Saves the AnnData object and PyTorch Geometric graph for a sample.

    Parameters:
    ----------
    adata : AnnData
        The AnnData object containing spatial and gene expression data.
    graph : torch_geometric.data.Data
        The PyTorch Geometric Data object representing the graph.
    output_dir : str
        Directory to save the files.
    sample_name : str
        Name of the sample (used for filenames).

    Raises:
    ------
    ValueError
        If `adata` or `graph` is None.

    Returns:
    -------
    None
    """
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # some checks
    if adata is None:
        raise ValueError("The `adata` object is None. Please provide a valid AnnData object.")
    if graph is None:
        raise ValueError("The `graph` object is None. Please provide a valid PyTorch Geometric Data object.")
    
    # save the AnnData object
    adata_file = os.path.join(output_dir, f"{sample_name}.h5ad")
    adata.write(adata_file)
    log.info(f"Saved AnnData object to {adata_file}")
    
    # save the PyTorch Geometric graph
    log.info(f"Creating graph for sample: {sample_name}")
    graph_file = os.path.join(output_dir, f"{sample_name}_graph.pt")
    torch.save(graph, graph_file)
    log.info(f"Saved graph to {graph_file}")

def preprocess_sample(adata, min_cells, min_genes, n_pca_components=50):
    """
    Preprocesses a single AnnData object by applying filtering, normalization, and dimensionality reduction steps.

    Parameters:
    ----------
    adata : AnnData
        The AnnData object containing spatial and gene expression data.
    min_cells : int
        The minimum number of cells in which a gene must be expressed to retain the gene.
    min_genes : int
        The minimum number of genes that must be expressed in a cell to retain the cell.
    n_pca_components : int, optional
        The number of PCA components to retain during dimensionality reduction. Default is 50.

    Returns:
    -------
    None
        The function modifies the input AnnData object in place.
    """
    total_genes = adata.n_vars
    min_genes_dynamic = max(min_genes, int(total_genes * 0.01))

    sc.pp.filter_genes(adata, min_cells=min_cells)              # filter genes expressed in less than min_cells cells
    sc.pp.filter_cells(adata, min_genes=min_genes_dynamic)      # filter cells with less than min_genes genes expressed
    sc.pp.normalize_total(adata, target_sum=1e5)                # normalize gene expression values with a target sum per cell of 1e5
    sc.pp.log1p(adata)                                          # log transform gene expression values (log1p(x) = log(x+1))
    sc.pp.scale(adata)                                          # scale gene expression values to unit variance and zero mean
    sc.pp.pca(adata, n_comps=n_pca_components)                  # perform PCA with n_pca_components components

def euclid_dist(t1, t2):
    """
    Computes the Euclidean distance between two vectors.

    Parameters:
    ----------
    t1 : np.ndarray
        The first vector.
    t2 : np.ndarray
        The second vector.

    Returns:
    -------
    float
        The Euclidean distance between the two vectors.
    """
    return np.linalg.norm(t1 - t2)

def create_graph(adata, sample_name, method, n_neighbors):
    """
    Creates a graph representation of spatial data from an AnnData object.

    Parameters:
    ----------
    adata : AnnData
        The AnnData object containing spatial and gene expression data.
    sample_name : str
        The name of the sample (determined from filenames).
    method : str, optional
        The method to use for graph construction. Options are "knn" or "pairwise".
        - "knn": Constructs a k-nearest neighbors graph based on spatial distances.
        - "pairwise": Constructs a fully connected graph with edge weights computed using a Gaussian kernel.
    n_neighbors : int, optional
        The number of nearest neighbors to consider for the "knn" method.

    Returns:
    -------
    torch_geometric.data.Data
        A PyTorch Geometric Data object containing the graph representation of the spatial data.

    Raises:
    ------
    ValueError
        If the AnnData object does not contain required attributes (`spatial` or `X_pca`).
        If an invalid method is provided.

    Notes:
    -----
    - For "knn":
        - Constructs a k-nearest neighbors graph based on spatial distances.
        - The adjacency matrix is symmetrized to ensure bidirectional edges.
        - The `edge_index` is derived from the adjacency matrix in COO (coordinate) format.
    - For "pairwise":
        - Constructs a fully connected graph where edge weights are computed using a Gaussian kernel:
          `weight = exp(-distance^2 / (2 * l^2))`.
        - The `edge_weight` attribute contains the computed edge weights.
    - The `x` attribute in the returned graph corresponds to the PCA-transformed features stored in `adata.obsm["X_pca"]`.
    - The adjacency matrix is stored in `adata.obsm['adj']` for both methods.
    """
    # some checks
    if "spatial" not in adata.obsm:
        raise ValueError("The AnnData object does not contain 'spatial' coordinates in `obsm`.")
    if "X_pca" not in adata.obsm:
        raise ValueError("The AnnData object does not contain PCA-transformed features in `obsm['X_pca']`.")

    if method == "knn":
        position = adata.obsm['spatial']
        
        # cKDTree for efficient k-NN search
        tree = cKDTree(position)
        _, indices = tree.query(position, k=n_neighbors + 1)  # +1 to include the point itself
        
        # create adjacency matrix
        adj = np.zeros((position.shape[0], position.shape[0]), dtype=np.int32)
        for i, neighbors in enumerate(indices):
            adj[i, neighbors[1:]] = 1  # skip the first neighbor (itself)
        
        # symmetrize adjacency matrix
        adj = np.maximum(adj, adj.T)
        adata.obsm['adj'] = adj

        # convert adjacency matrix to COO format
        edge_index = np.array(np.nonzero(adj))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float)

        return Data(
            x=torch.tensor(adata.obsm["X_pca"].copy(), dtype=torch.float), 
            edge_index=edge_index,
            edge_weight=edge_weight,
            sample_name=sample_name
        )

    elif method == "pairwise":
        X = adata.obsm["spatial"]
        n = X.shape[0]

        # calculate adjacency matrix based on euclidean distance between all pairs of cells
        adj = np.empty((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                adj[i][j] = euclid_dist(X[i], X[j])

        # calculate the edge weights based on the euclidean distances
        l = np.mean(adj)
        adj_exp = np.exp(-1 * (adj**2) / (2 * (l**2)))

        adata.obsm['adj'] = adj_exp

        # convert adjacency matrix to COO format for edge index and edge attributes
        edge_index = np.array(np.nonzero(adj))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(adj_exp[edge_index[0], edge_index[1]], dtype=torch.float)

        return Data(
            x=torch.tensor(adata.obsm["X_pca"].copy(), dtype=torch.float), 
            edge_index=edge_index, 
            edge_weight=edge_weight,
            sample_name=sample_name
        )
        
    else:
        raise ValueError("Invalid method. Choose between 'knn' and 'pairwise'")


class SpatialOmicsDataset(Dataset):
    """
    A custom PyTorch Geometric dataset for spatial omics data.

    Attributes:
    ----------
    samples : list
        A list of PyTorch Geometric Data objects representing the dataset.

    Methods:
    -------
    len() -> int
        Returns the number of samples in the dataset.
    get(idx: int) -> Data
        Returns the sample at the specified index.
    """
    def __init__(self, samples: dict):
        """
        Initializes the SpatialOmicsDataset.

        Parameters:
        ----------
        samples : dict
            A dictionary where the keys are sample names and the values are PyTorch Geometric Data objects.
        """
        super().__init__()
        self.samples = list(samples.values())

    def len(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.samples)

    def get(self, idx: int) -> Data:
        """
        Returns the sample at the specified index.

        Parameters:
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns:
        -------
        torch_geometric.data.Data
            The PyTorch Geometric Data object at the specified index.
        """
        return self.samples[idx]