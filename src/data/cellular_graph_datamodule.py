"""
Cellular Graph Data Module.

This module defines the `CellularGraphDataset` and `CellularGraphDataModule` classes, which are designed for handling spatial
omics data represented as graphs. These classes provide functionality for loading, preprocessing, and managing datasets of
graphs constructed from spatial omics data.

Features:
---------
- Supports preprocessing of spatial omics graphs stored in `.gpt` files.
- Constructs k-hop subgraphs for graph-based learning tasks.
- Provides train, validation, and test dataloaders for PyTorch Geometric models.
- Handles class balancing and sampling strategies for subgraph generation.
- Supports distributed training with PyTorch Lightning.

Classes:
--------
- CellularGraphDataset: A PyTorch Geometric Dataset for managing cellular graphs.
- CellularGraphDataModule: A PyTorch Lightning DataModule for managing the dataset and dataloaders.

Usage:
------
>>> from src.data.cellular_graph_datamodule import CellularGraphDataset, CellularGraphDataModule
>>> dataset = CellularGraphDataset(regions, graph_dir, json_path, subgraph_size, batch_size, num_iterations, training)
>>> datamodule = CellularGraphDataModule(mode, data_dir, processed_dir, batch_size, num_workers, ...)
>>> datamodule.prepare_data()
>>> datamodule.setup()
>>> train_loader = datamodule.train_dataloader()
"""

# TODO: add subgraph_radius_limit (see Lovro's data.py)


import copy
import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class CellularGraphDataset(Dataset):
    """
    A PyTorch Geometric Dataset for managing cellular graphs.

    This dataset handles the loading and preprocessing of cellular graphs, as well as the generation of k-hop subgraphs
    for graph-based learning tasks. It supports balanced sampling of subgraphs based on cell type frequencies and
    provides flexibility for training and evaluation modes.

    Attributes:
    -----------
    regions : list
        A list of region identifiers (e.g., sample names) that correspond to the dataset.
    graph_dir : str
        Path to the directory where graph files are stored.
    json_path : str
        Path to the directory containing JSON files for cell type mapping, frequency, and biomarkers.
    subgraph_size : int
        The size of each subgraph (k-hop).
    batch_size : int
        The batch size for subgraph sampling.
    num_iterations : int
        The number of iterations for subgraph sampling.
    training : bool
        Whether the dataset is used for training or evaluation.
    sampling_avoid_unassigned : bool
        Whether to avoid sampling unassigned cells during subgraph sampling.
    unassigned_cell_type : str
        The cell type to be considered as unassigned.
    sampling_freq : torch.Tensor
        Sampling frequency for each cell type, used for balanced subgraph sampling.
    """

    def __init__(
        self,
        regions: list,
        graph_dir: str,
        json_path: str,
        subgraph_size: int,
        batch_size: int,
        num_iterations: int,
        training: bool,
        node_features: list,
        edge_features: list,
        node_features_to_use: list,
        edge_features_to_use: list,
        sampling_avoid_unassigned: bool,
        unassigned_cell_type: str,
    ) -> None:
        """
        Initializes the CellularGraphDataset.

        Parameters:
        -----------
        regions : list
            A list of region identifiers (e.g., sample names) that correspond to the dataset.
        graph_dir : str
            Path to the directory where graph files are stored.
        json_path : str
            Path to the directory containing JSON files for cell type mapping, frequency, and biomarkers.
        subgraph_size : int
            The size of each subgraph (k-hop).
        batch_size : int
            The batch size for subgraph sampling.
        num_iterations : int
            The number of iterations for subgraph sampling.
        training : bool
            Whether the dataset is used for training or evaluation.
        node_features : list, optional
            List of node features to be included in the dataset.
        edge_features : list, optional
            List of edge features to be included in the dataset.
        node_features_to_use : list, optional
            List of node features to be used by the model.
        edge_features_to_use : list, optional
            List of edge features to be used by the model.
        sampling_avoid_unassigned : bool, optional
            Whether to avoid sampling unassigned cells during subgraph sampling.
        unassigned_cell_type : str, optional
            The cell type to be considered as unassigned.
        """
        super().__init__()
        self.training = training

        # initinalize regions and graph location
        self.regions = regions
        self.graph_dir = graph_dir

        # initialize subgraph parameters
        self.subgraph_size = subgraph_size
        self.num_iterations = num_iterations
        self.batch_size = batch_size

        # initinalize cell type mapping, frequency, and biomarkers
        ct_mapping_path = os.path.join(json_path, "cell_type_mapping.json")
        ct_freq_path = os.path.join(json_path, "cell_type_freq.json")
        biomarkers_path = os.path.join(json_path, "biomarkers.json")
        with open(ct_mapping_path) as f:
            self.cell_type_mapping = json.load(f)
        with open(ct_freq_path) as f:
            self.cell_type_freq = json.load(f)
        with open(biomarkers_path) as f:
            self.biomarkers = json.load(f)

        # sampling frequency for each cell type for subgraph sampling
        self.sampling_freq = {
            self.cell_type_mapping[ct]: 1.0 / np.log(self.cell_type_freq[ct] + 1 + 1e-5)
            for ct in self.cell_type_mapping
        }
        self.sampling_freq = torch.from_numpy(
            np.array([self.sampling_freq[i] for i in range(len(self.sampling_freq))])
        )

        # optionally avoid sampling unassigned cells
        self.unassigned_cell_type = unassigned_cell_type
        if sampling_avoid_unassigned and unassigned_cell_type in self.cell_type_mapping:
            self.sampling_freq[self.cell_type_mapping[unassigned_cell_type]] = 0.0
        if "Unknown" in self.cell_type_mapping.keys():
            self.sampling_freq[self.cell_type_mapping["Unknown"]] = 0.0

        # initialize node features and edge features
        self.node_features = node_features
        self.edge_features = edge_features
        if "cell_type" in self.node_features:
            assert (
                self.node_features.index("cell_type") == 0
            ), "cell_type must be the first node feature"
        if "edge_type" in self.edge_features:
            assert (
                self.edge_features.index("edge_type") == 0
            ), "edge_type must be the first edge feature"
        self.node_features_to_use = node_features_to_use
        self.edge_features_to_use = edge_features_to_use

        # get the feature names
        self.node_feature_names = self.get_feature_names(node_features)
        self.edge_feature_names = self.get_feature_names(edge_features)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        --------
        int
            If training, it returns the number of subgraphs to be sampled (which batch_size * num_iterations);
            otherwise, it returns the number of regions in the dataset.
        """
        if self.training:
            # return self.batch_size * self.num_iterations
            return self.batch_size
        else:
            return len(self.regions)

    def pick_center_node(self, graph: Data) -> int:
        """
        Randomly selects a center node from the graph based on the sampling frequency
        to ensure balanced sampling of cell types.

        Parameters:
        -----------
        graph : Data
            The graph from which to select the center node.

        Returns:
        --------
        int
            The index of the selected center node.
        """
        # compute sampling frequencies
        cell_types = graph.x[:, 0].long()
        freq = self.sampling_freq.gather(0, cell_types)
        freq = freq / freq.sum()

        # randomly select a center node
        center_node_ind = int(np.random.choice(np.arange(len(freq)), p=freq.cpu().data.numpy()))
        return center_node_ind

    def get_feature_names(self, features: list[str]) -> list[str]:
        """ """
        feature_names = []
        for feature in features:
            if feature in ["distance", "cell_type", "edge_type", "size"]:
                # feature "cell_type", "edge_type" will be a single integer indice
                # feature "distance", "size" will be a single float value
                feature_names.append(feature)
            elif feature == "center_coord":
                # feature "center_coord" will be a tuple of two float values
                feature_names.extend(["center_coord-x", "center_coord-y"])
            elif feature == "biomarker_expression":
                # feature "biomarker_expression" will contain a list of biomarker expression values
                feature_names.extend(["biomarker_expression-%s" % bm for bm in self.biomarkers])
            elif feature == "neighborhood_composition":
                # feature "neighborhood_composition" will contain a composition vector of the immediate neighbors
                # the vector will have the same length as the number of unique cell types
                feature_names.extend(
                    [
                        "neighborhood_composition-%s" % ct
                        for ct in sorted(
                            self.cell_type_mapping.keys(), key=lambda x: self.cell_type_mapping[x]
                        )
                    ]
                )
            else:
                log.warning("Using additional feature: %s" % feature)
                feature_names.append(feature)
        return feature_names

    def mask_features(self, graph: Data) -> None:
        """
        Masks the node and edge features that should not be used by the model.

        Parameters:
        -----------
        graph : Data
            The graph to be masked.

        Side Effects:
        -----------
        - Modifies the node features and edge features of the graph in place.
        - Sets the features that should not be used by the model to zero.
        """
        # mask node features that should not be used by the model
        for name in self.node_feature_names:
            if all(not name.startswith(feature) for feature in self.node_features_to_use):
                graph.x[:, self.node_feature_names.index(name)] = 0

        # mask edge features that should not be used by the model
        for name in self.edge_feature_names:
            if all(not name.startswith(feature) for feature in self.edge_features_to_use):
                graph.edge_attr[:, self.edge_feature_names.index(name)] = 0

    def get_graph(self, idx: int) -> Data:
        """
        Loads and returns the graph data for the region at the specified index.

        Parameters:
        -----------
        idx : int
            The index of the region to retrieve.

        Returns:
        --------
        torch_geometric.data.Data
            A PyTorch Geometric `Data` object representing the graph for the specified region.

        Raises:
        -------
        FileNotFoundError
            If the graph file for the specified region does not exist in `graph_dir`.
        """
        region = self.regions[idx]
        graph_path = os.path.join(self.graph_dir, f"{region}.0.gpt")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        graph = torch.load(graph_path, weights_only=False)  # nosec B614
        self.mask_features(graph)
        return graph

    def save_all_subgraphs_to_chunk(self):
        """
        Save all n-hop subgraphs for all regions to chunk files (one file per region).
        """
        for idx, region in tqdm(enumerate(self.regions)):
            # check if the subgraph file already exists
            subgraph_path = os.path.join(self.graph_dir, region + f".{self.subgraph_size}-hop.gpt")
            if os.path.exists(subgraph_path):
                continue

            # if not, load the graph and sample all subgraphs
            subgraphs = self.get_all_subgraphs(idx)
            torch.save(subgraphs, subgraph_path)

    def load_subgraph_from_chunk(self, region_idx: int, center_node_ind: int) -> Data:
        """
        Loads a specific subgraph from disk (stored as a chunk file for the region).
        Falls back to computing it on the fly if the file is missing.

        Parameters:
        -----------
        region_idx : int
            Index of the region in the dataset.
        center_node_ind : int
            Index of the center node in the region's graph.

        Returns:
        --------
        Data
            The k-hop subgraph centered on the specified node.
        """
        region = self.regions[region_idx]
        subgraph_path = os.path.join(self.graph_dir, f"{region}.{self.subgraph_size}-hop.gpt")

        if not os.path.exists(subgraph_path):
            raise FileNotFoundError(f"Subgraph file not found: {subgraph_path}.")

        subgraphs = torch.load(subgraph_path, weights_only=False)  # nosec B614
        return subgraphs[center_node_ind]

    def get_all_subgraphs(self, idx: int) -> list:
        """
        Loads and returns a list of all subgraphs for the region at the specified index.

        Parameters:
        -----------
        idx : int
            The index of the region to retrieve.

        Returns:
        --------
        list
            A list of PyTorch Geometric `Data` objects representing the subgraphs for the specified region.
        """
        # get full graph to sample from
        graph = self.get_graph(idx)

        # load the subgraph from the chunk file if it exists
        path = os.path.join(self.graph_dir, f"{graph.region_id}.{self.subgraph_size}-hop.gpt")
        if os.path.exists(path):
            path = os.path.join(self.graph_dir, f"{graph.region_id}.{self.subgraph_size}-hop.gpt")
            return torch.load(path, weights_only=False)  # nosec B614

        # sample and build all subgraphs
        subgraphs = []
        for node in range(graph.num_nodes):
            center_node_ind = int(node)

            # sample a k-hop subgraph around the center node
            node_idx, edge_index, _, edge_mask = k_hop_subgraph(
                center_node_ind,
                self.subgraph_size,
                graph.edge_index,
                relabel_nodes=True,
                num_nodes=graph.num_nodes,
            )

            # create the subgraph and append it to the list of subgraphs
            subgraph = Data(
                x=graph.x[node_idx],
                edge_index=edge_index,
                edge_attr=graph.edge_attr[edge_mask],
                region_id=graph.region_id,
                y=graph.y if graph.y is not None else None,
                w=graph.w if graph.w is not None else None,
                original_center_node=center_node_ind,
            )
            subgraphs.append(subgraph)

        return subgraphs

    def get_subgraph(self, idx: int) -> Data:
        """
        Samples and returns a single k-hop subgraph for a randomly selected region.

        Parameters:
        -----------
        idx : int
            The index of the subgraph to retrieve (used for iteration).

        Returns:
        --------
        torch_geometric.data.Data
            A PyTorch Geometric `Data` object representing the sampled subgraph.
        """
        # randomly select a region and load its graph
        region_idx = np.random.randint(0, len(self.regions))
        graph = self.get_graph(region_idx)

        # randomly select a center node from the graph and load or build a k-hop subgraph
        center_node_ind = self.pick_center_node(graph)

        # loading subgraph is slower in this case so not used
        # load the subgraph from the chunk file if it exists
        # path = os.path.join(self.graph_dir, f"{graph.region_id}.{self.subgraph_size}-hop.gpt")
        # if os.path.exists(path):
        #    return self.load_subgraph_from_chunk(region_idx, center_node_ind)

        # build a k-hop subgraph around the center node
        node_idx, edge_index, _, edge_mask = k_hop_subgraph(
            center_node_ind,
            self.subgraph_size,
            graph.edge_index,
            relabel_nodes=True,
            num_nodes=graph.num_nodes,
        )

        return Data(
            x=graph.x[node_idx],
            edge_index=edge_index,
            edge_attr=graph.edge_attr[edge_mask],
            region_id=graph.region_id,
            y=graph.y if graph.y is not None else None,
            w=graph.w if graph.w is not None else None,
            original_center_node=center_node_ind,
        )

    def __getitem__(self, idx: int) -> Data:
        """
        Retrieves data based on the mode (training or evaluation).

        Parameters:
        -----------
        idx : int
            The index of the data to retrieve.

        Returns:
        --------
        torch_geometric.data.Data or list
            If training, returns a single k-hop subgraph. If evaluation, returns all subgraphs for the region.
        """
        if self.training:
            return self.get_subgraph(idx)
        else:
            subgraphs = self.get_all_subgraphs(idx)
            return Batch.from_data_list(subgraphs)


class CellularGraphDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for managing cellular graph datasets.

    This class handles the loading, preprocessing, and management of cellular graph datasets.
    It provides train, validation, and test dataloaders for PyTorch Geometric models and supports
    distributed training.

    Attributes:
    -----------
    mode : str
        Mode of operation (e.g., "pretraining", "finetuning", "evaluation").
    data_dir : str
        Directory containing raw spatial omics data files.
    processed_dir : str
        Directory to save/load preprocessed graphs.
    batch_size : int
        Batch size for dataloaders.
    num_workers : int
        Number of workers for data loading.
    pin_memory : bool
        Whether to pin memory for DataLoader.
    json_path : str
        Path to the JSON file containing cell type mapping and frequency.
    subgraph_size : int
        Size of each subgraph (k-hop).
    num_iterations : int
        Number of iterations for subgraph sampling.
    sampling_avoid_unassigned : bool
        Whether to avoid sampling unassigned cells during subgraph sampling.
    unassigned_cell_type : str
        Cell type to be considered as unassigned.
    graph_tasks : list
        List of tasks for which the graph is constructed.
    redo_preprocess : bool
        Whether to redo preprocessing even if preprocessed data exists.
    seed : int
        Random seed for reproducibility.
    batch_size_per_device : int
        Batch size for dataloaders, adjusted for distributed training.
    dataset : CellularGraphDataset
        Dataset object containing graphs and metadata.
    train_dataset : CellularGraphDataset
        Dataset used for training.
    val_dataset : CellularGraphDataset
        Dataset used for validation.
    test_dataset : CellularGraphDataset
        Dataset used for testing.
    """

    def __init__(
        self,
        mode: str,
        data_dir: str,
        processed_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        json_path: str,
        subgraph_size: int,
        num_iterations: int,
        node_features: list,
        edge_features: list,
        node_features_to_use: list,
        edge_features_to_use: list,
        sampling_avoid_unassigned: bool,
        unassigned_cell_type: str,
        graph_tasks: list,
        redo_preprocess: bool,
        seed: int,
    ) -> None:
        """
        Initialize the CellularGraphDataModule.

        Parameters:
        -----------
        mode : str
            Mode of operation (e.g., "pretraining", "finetuning", "evaluation").
        data_dir : str
            Directory containing raw spatial omics data files.
        processed_dir : str
            Directory to save/load preprocessed graphs.
        batch_size : int
            Batch size for dataloaders.
        num_workers : int
            Number of workers for data loading.
        pin_memory : bool
            Whether to pin memory for DataLoader.
        json_path : str
            Path to the JSON file containing cell type mapping and frequency.
        subgraph_size : int
            Size of each subgraph (k-hop).
        num_iterations : int
            Number of iterations for subgraph sampling.
        node_features : list
            List of node features to be included in the dataset.
        edge_features : list
            List of edge features to be included in the dataset.
        node_features_to_use : list
            List of node features to be used by the model.
        edge_features_to_use : list
            List of edge features to be used by the model.
        sampling_avoid_unassigned : bool
            Whether to avoid sampling unassigned cells during subgraph sampling.
        unassigned_cell_type : str
            Cell type to be considered as unassigned.
        graph_tasks : list
            List of tasks for which the graph is constructed.
        redo_preprocess : bool
            Whether to redo preprocessing even if preprocessed data exists.
        seed : int
            Random seed for reproducibility.

        Raises:
        -------
        ValueError
            If the mode is not one of "pretraining", "finetuning", or "evaluation".
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        if mode in ["pretraining", "finetuning"]:
            self.hparams.training = True
        elif mode in ["evaluation"]:
            self.hparams.training = False
        else:
            raise ValueError(
                f"Invalid mode {mode}. Supported modes are 'pretraining', 'finetuning', and 'evaluation'."
            )

        self.batch_size_per_device = batch_size

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # ensure the processed directory exists
        os.makedirs(self.hparams.processed_dir, exist_ok=True)

    def prepare_data(self) -> None:
        """
        Verify that the raw data files exist.

        This method checks if the directory containing raw data exists.
        If the directory does not exist, it raises a FileNotFoundError. Lightning ensures
        that `self.prepare_data()` is called only within a single process on CPU.

        Raises:
        -------
        FileNotFoundError
            If the raw data directory does not exist.
        """
        if not os.path.exists(self.hparams.data_dir):
            raise FileNotFoundError(
                f"Data directory {self.hparams.data_dir} does not exist. Please provide the data."
            )
        else:
            log.info(
                f"Data directory {self.hparams.data_dir} exists. Proceeding with preprocessing or loading the data."
            )

    def process_graph(
        self, file_path: str, graph_tasks: list, targets: pd.DataFrame, class_weights: torch.Tensor
    ) -> None:
        """
        Processes a single cellular graph.

        This method reads a cellular graph from a `.gpt` file, appends target values
        and class weights to the graph, and saves the processed graph to disk.

        Parameters:
        -----------
        file_path : str
            The path to the `.gpt` file containing the raw spatial omics data for the sample.
        graph_tasks : list
            A list of tasks for which the graph is constructed. Each task corresponds to a column
            in the targets DataFrame.
        targets : pd.DataFrame
            A DataFrame containing the target values for each sample. The DataFrame should have
            a column named "REGION_ID" that matches the sample name in the `.gpt` file.
        class_weights : torch.Tensor
            A tensor containing the class weights for each task. This is used to balance the dataset
            during training.

        Returns:
        --------
        str
            The name of the processed sample (derived from the file name without the extension).

        Side Effects:
        -------------
        - Saves the processed graph to the directory specified in `self.hparams.processed_dir`.

        Raises:
        -------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If a task is not found in the targets DataFrame or contains non-numeric values.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist. Please provide the data.")
        else:
            log.info(f"Processing graph {file_path}.")

        # read the sample
        graph = torch.load(file_path, weights_only=False)  # nosec B614

        # append the targets and weights to the graph
        graph.y = torch.zeros((len(graph_tasks)), dtype=torch.float32)
        graph.w = torch.zeros((len(graph_tasks)), dtype=torch.float32)
        for i, task in enumerate(graph_tasks):
            if task not in targets.columns:
                raise ValueError(f"Task {task} not found in targets.")
            graph.y[i] = torch.tensor(
                targets.loc[targets["REGION_ID"] == graph.region_id, task].values
            )
            graph.y[i] = graph.y[i].to(torch.float32)
            graph.w[i] = class_weights[task][graph.y[i].item()]

        # save processed graph
        graph_path = os.path.join(self.hparams.processed_dir, f"{graph.region_id}.0.gpt")
        torch.save(graph, graph_path)
        log.info(f"Saved processed graph to {graph_path}.")

        return graph.region_id

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load and preprocess data, and split into train, validation, and test datasets.

        This method loads preprocessed graphs from disk if they exist. If not, it processes
        the graphs, and saves the processed graphs to disk. It also splits the dataset into
        train, validation, and test datasets.

        Parameters:
        -----------
        stage : str, optional
            The stage of the setup process (e.g., "fit", "test"). Default is None.

        Raises:
        -------
        RuntimeError
            If the batch size is not divisible by the number of devices in distributed training.
        FileNotFoundError
            If the targets file or raw data files are missing.
        ValueError
            If a task in `graph_tasks` is not found in the targets DataFrame or contains invalid values.
        """
        # divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and process datasets only if not loaded already
        if not self.dataset:
            processed_file = os.path.join(self.hparams.processed_dir, "dataset.pt")

            if os.path.exists(processed_file) and not self.hparams.redo_preprocess:
                # load preprocessed graphs and not redo preprocessing
                self.dataset = torch.load(processed_file, weights_only=False)  # nosec B614
                log.info(
                    f"Loaded preprocessed graphs from {os.path.join(self.hparams.processed_dir, processed_file)}."
                )
            else:
                # preprocess data and save to disk
                log.info(
                    f"Preprocessing data from {self.hparams.data_dir} and saving to {self.hparams.processed_dir}."
                )
                file_paths = [
                    os.path.join(self.hparams.data_dir, f)
                    for f in os.listdir(self.hparams.data_dir)
                    if f.endswith(".gpt")
                ]

                # read the targets
                targets_path = os.path.join(self.hparams.data_dir, "targets.csv")
                if not os.path.exists(targets_path):
                    raise FileNotFoundError(
                        f"File {targets_path} does not exist. Please provide the targets."
                    )
                targets = pd.read_csv(targets_path)

                # calculate class weights to balance the dataset
                class_label_weights = {}
                for i, task in enumerate(self.hparams.graph_tasks):
                    if task not in targets.columns:
                        raise ValueError(f"Task {task} not found in targets.")
                    ar = list(targets[task])
                    valid_vals = [_y for _y in ar if _y == _y]
                    unique_vals = set(valid_vals)
                    if not all(v.__class__ in [int, float] for v in unique_vals):
                        raise ValueError(f"Task {task} contains non-numeric values.")
                    val_counts = {_y: valid_vals.count(_y) for _y in unique_vals}
                    max_count = max(val_counts.values())
                    class_label_weights[task] = {
                        _y: max_count / val_counts[_y] for _y in unique_vals
                    }

                # preprocess graphs (optionally in parallel)
                if self.hparams.num_workers <= 1:
                    log.info("Preprocessing graphs in serial.")
                    regions = []
                    for file_path in file_paths:
                        graph_region = self.process_graph(
                            file_path, self.hparams.graph_tasks, targets, class_label_weights
                        )
                        regions.append(graph_region)
                else:
                    log.info("Preprocessing samples in parallel.")
                    # use joblib to parallelize the preprocessing
                    regions = Parallel(n_jobs=self.hparams.num_workers)(
                        delayed(function=self.process_graph)(
                            file_path, self.hparams.graph_tasks, targets, class_label_weights
                        )
                        for file_path in file_paths
                    )
                # save preprocessed graphs
                self.dataset = CellularGraphDataset(
                    regions=regions,
                    graph_dir=self.hparams.processed_dir,
                    json_path=self.hparams.json_path,
                    subgraph_size=self.hparams.subgraph_size,
                    batch_size=self.hparams.batch_size,
                    num_iterations=self.hparams.num_iterations,
                    training=self.hparams.training,
                    node_features=self.hparams.node_features,
                    edge_features=self.hparams.edge_features,
                    node_features_to_use=self.hparams.node_features_to_use,
                    edge_features_to_use=self.hparams.edge_features_to_use,
                    sampling_avoid_unassigned=self.hparams.sampling_avoid_unassigned,
                    unassigned_cell_type=self.hparams.unassigned_cell_type,
                )
                torch.save(self.dataset, os.path.join(self.hparams.processed_dir, "dataset.pt"))
                log.info(f"Saved preprocessed graphs to {processed_file}. Finished preprocessing.")

                # save all subgraphs to chunk files if they don't exist yet
                log.info(f"Saving all subgraphs to chunk files in {self.hparams.processed_dir}.")
                self.dataset.save_all_subgraphs_to_chunk()
                log.info(f"Saved all subgraphs to chunk files in {self.hparams.processed_dir}.")

        # set seeds for reproducibility
        np.random.seed(self.hparams.seed)
        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)
        torch.cuda.manual_seed_all(self.hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # load regions for training and validation
        train_region_path = os.path.join(self.hparams.json_path, "train_regions.json")
        val_region_path = os.path.join(self.hparams.json_path, "valid_regions.json")
        with open(train_region_path) as f:
            train_regions = json.load(f)
        with open(val_region_path) as f:
            val_test_regions = json.load(f)

        # set the train dataset (use all data for pretraining, use training regions for finetuning)
        # during evaluation, no training data is used
        if self.hparams.mode == "pretraining":
            self.train_dataset = self.dataset
        elif self.hparams.mode == "finetuning":
            self.train_dataset = copy.deepcopy(self.dataset)
            self.train_dataset.regions = train_regions
            self.train_dataset.training = True
        else:
            self.train_dataset = None

        # set the validation and test datasets (split into 50/50)
        # during pretraining, no validation data is used
        if self.hparams.mode == "pretraining":
            self.val_dataset = None
            self.test_dataset = None
        else:
            # split the validation and test regions into 50/50
            val_size = int(0.5 * len(val_test_regions))
            val_regions = random.sample(val_test_regions, val_size)
            test_regions = [r for r in val_test_regions if r not in val_regions]

            # set the validation dataset
            self.val_dataset = copy.deepcopy(self.dataset)
            self.val_dataset.regions = val_regions
            self.val_dataset.training = False
            self.val_dataset.batch_size = 1

            # set the test dataset
            self.test_dataset = copy.deepcopy(self.dataset)
            self.test_dataset.regions = test_regions
            self.test_dataset.training = False
            self.test_dataset.batch_size = 1

    def worker_init_fn(self, worker_id):
        base_seed = torch.initial_seed() % 2**32
        np.random.seed(base_seed + worker_id)
        random.seed(base_seed + worker_id)

    def train_dataloader(self) -> DataLoader:
        """
        Return the train dataloader.

        Returns:
        -------
        DataLoader
            A PyTorch Geometric DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            worker_init_fn=self.worker_init_fn,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the validation dataloader.

        Returns:
        -------
        DataLoader
            A PyTorch Geometric DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
            persistent_workers=False,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the test dataloader.

        Returns:
        -------
        DataLoader
            A PyTorch Geometric DataLoader for the test dataset.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
            persistent_workers=False,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = CellularGraphDataModule()
