"""
Spatial Omics Data Module.

This module defines the `SpatialOmicsDataModule` class, which is a PyTorch Lightning DataModule
designed for handling spatial omics data. It provides functionality for loading, preprocessing,
and managing datasets of graphs constructed from spatial omics data.

Features:
---------
- Supports preprocessing of raw spatial omics data stored in `.h5ad` files.
- Constructs graphs using methods such as k-nearest neighbors (k-NN).
- Saves and loads preprocessed graphs to/from disk for efficient reuse.
- Provides train, validation, and test dataloaders for PyTorch Geometric models.

Classes:
--------
- SpatialOmicsDataModule: A Lightning DataModule for spatial omics data.

Usage:
------
>>> from src.data.spatial_omics_datamodule import SpatialOmicsDataModule
>>> datamodule = SpatialOmicsDataModule(data_dir="data/domain/raw", processed_dir="data/domain/processed")
>>> datamodule.prepare_data()
>>> datamodule.setup()
>>> train_loader = datamodule.train_dataloader()
"""

import gc
import os
from typing import Any, Dict, Optional

import scanpy as sc
import torch
from joblib import Parallel, delayed
from lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from src.utils import RankedLogger
from src.utils.preprocess_helpers import (
    SpatialOmicsDataset,
    create_graph,
    preprocess_sample,
    save_sample,
)

log = RankedLogger(__name__, rank_zero_only=True)


class SpatialOmicsDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for spatial omics data.

    This class handles the loading, preprocessing, and management of spatial omics datasets.
    It supports graph construction from spatial omics data and provides train, validation, and
    test dataloaders for PyTorch Geometric models.

    Attributes:
    -----------
    batch_size_per_device : int
        Batch size for dataloaders, adjusted for distributed training.
    dataset : SpatialOmicsDataset
        Dataset object containing graphs and metadata.
    train_dataset : SpatialOmicsDataset
        Dataset used for training.
    val_dataset : SpatialOmicsDataset
        Dataset used for validation.
    test_dataset : SpatialOmicsDataset
        Dataset used for testing.
    """

    def __init__(
        self,
        data_dir: str,
        processed_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        min_cells: int,
        min_genes: int,
        augmentation_mode: str,
        lambda_param: float,
        sigma_param: float,
        n_pca_components: int,
        graph_method: str,
        n_neighbors: int,
        redo_preprocess: bool,
        seed: int,
    ) -> None:
        """
        Initialize the SpatialOmicsDataModule.

        Parameters:
        -----------
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
        min_cells : int
            Minimum number of cells required for preprocessing.
        min_genes : int
            Minimum number of genes required for preprocessing.
        augmentation_mode : str
            Augmentation mode for preprocessing.
        lambda_param : float
            Lambda parameter for pseudo batch effect.
        sigma_param : float
            Sigma parameter for pseudo batch effect.
        n_pca_components : int
            Number of PCA components for dimensionality reduction.
        graph_method : str
            Method for graph construction ("knn" or "pairwise").
        n_neighbors : int
            Number of neighbors for k-NN graph construction.
        redo_preprocess : bool
            Whether to redo preprocessing even if preprocessed data exists.
        seed : int
            Random seed for reproducibility.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

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

        This method checks if the directory containing raw spatial omics data exists.
        If the directory does not exist, it raises a FileNotFoundError. Lightning ensures
        that `self.prepare_data()` is called only within a single process on CPU.

        Raises:
        -------
        FileNotFoundError
            If the raw data directory does not exist.
        """
        if not os.path.exists(self.hparams.data_dir):
            raise FileNotFoundError(
                f"Data directory {self.hparams.data_dir} does not exist. Please download the data."
            )
        else:
            log.info(
                f"Data directory {self.hparams.data_dir} exists. Proceeding with preprocessing or loading the data."
            )

    def process_sample(self, file_path: str) -> None:
        """
        Process a single spatial omics sample.

        This function reads a spatial omics sample from an `.h5ad` file, applies preprocessing steps
        (e.g., filtering, normalization, PCA, and augmentation), constructs a graph representation
        of the sample, and saves the preprocessed data and graph to disk.

        Parameters:
        -----------
        file_path : str
            The path to the `.h5ad` file containing the raw spatial omics data for the sample.

        Returns:
        --------
        str
            The name of the processed sample (derived from the file name without the extension).

        Side Effects:
        -------------
        - Saves the preprocessed AnnData object and its corresponding graph to the directory specified
        in `self.hparams.processed_dir`.
        - Frees memory by deleting intermediate objects and invoking garbage collection.

        Notes:
        ------
        - If the sample has already been preprocessed (indicated by the presence of "X_pca" in
        `adata.obsm`), preprocessing is skipped.
        - The graph is constructed using the method and parameters specified in `self.hparams`.

        Raises:
        -------
        FileNotFoundError
            If the specified file does not exist.
        """
        adata = sc.read_h5ad(file_path)
        sample_name = os.path.splitext(os.path.basename(file_path))[0]

        if "X_pca" in adata.obsm:
            log.info(f"Sample {sample_name} is already preprocessed. Skipping preprocessing.")
        else:
            preprocess_sample(
                adata,
                sample_name,
                min_cells=self.hparams.min_cells,
                min_genes=self.hparams.min_genes,
                n_pca_components=self.hparams.n_pca_components,
                augmentation_mode=self.hparams.augmentation_mode,
                lambda_param=self.hparams.lambda_param,
                sigma_param=self.hparams.sigma_param,
            )
        graph = create_graph(
            adata,
            sample_name=sample_name,
            method=self.hparams.graph_method,
            n_neighbors=self.hparams.n_neighbors,
        )

        # rescale STARmap positions to match the other samples in domain123 (median of ~45)
        if graph.sample_name.startswith("STARmap"):
            log.info(
                f"Rescaling STARmap sample {sample_name} to match other samples' neighbor distances."
            )
            edge_index = graph.edge_index
            pos = graph.position

            src, dst = edge_index
            dists = torch.norm(pos[src] - pos[dst], dim=1)
            valid_dists = dists[dists > 0]
            median_dist = valid_dists.median().item()

            scale = 35 / median_dist
            graph.position *= scale

        save_sample(adata, graph, self.hparams.processed_dir, sample_name)
        del adata
        del graph
        gc.collect()

        return sample_name

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load and preprocess spatial omics data.

        This method loads preprocessed graphs from disk if they exist. If not, it preprocesses
        the raw data, constructs graphs, and saves the preprocessed graphs to disk. It also
        splits the dataset into train, validation, and test datasets.

        Parameters:
        -----------
        stage : str, optional
            The stage of the setup process (e.g., "fit", "test"). Default is None.

        Raises:
        -------
        RuntimeError
            If the batch size is not divisible by the number of devices in distributed training.
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
                    if f.endswith(".h5ad")
                ]

                # preprocess samples and create graphs (optionally in parallel)
                if self.hparams.num_workers <= 1:
                    log.info("Preprocessing samples in serial.")
                    samples = []
                    for file_path in file_paths:
                        sample_name = self.process_sample(file_path)
                        samples.append(sample_name)
                else:
                    log.info("Preprocessing samples in parallel.")
                    # use joblib to parallelize the preprocessing
                    samples = Parallel(n_jobs=self.hparams.num_workers)(
                        delayed(function=self.process_sample)(file_path)
                        for file_path in file_paths
                    )
                # save preprocessed graphs
                self.dataset = SpatialOmicsDataset(
                    samples=samples, graph_dir=self.hparams.processed_dir
                )
                torch.save(self.dataset, os.path.join(self.hparams.processed_dir, "dataset.pt"))
                log.info(f"Saved preprocessed graphs to {processed_file}. Finished preprocessing.")

        # keep all data for training since no labels are used during training
        self.train_dataset = self.dataset

        # split data into validation (40) and test (60) sets (here, labels are used)
        generator = torch.Generator().manual_seed(self.hparams.seed)
        val_size = int(0.4 * len(self.dataset))
        test_size = len(self.dataset) - val_size
        self.val_dataset, self.test_dataset = random_split(
            self.dataset, [val_size, test_size], generator=generator
        )

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
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
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
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
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
    _ = SpatialOmicsDataModule()
