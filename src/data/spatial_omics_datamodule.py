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


from typing import Optional, Dict, Any
import os
import torch
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from src.utils.preprocess_helpers import (
    read_samples_into_dict, 
    preprocess_sample, 
    create_graph, 
    save_sample, 
    SpatialOmicsDataset
)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SpatialOmicsDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for spatial omics data.

    This class handles the loading, preprocessing, and management of spatial omics datasets.
    It supports graph construction from spatial omics data and provides train, validation, and
    test dataloaders for PyTorch Geometric models.

    Attributes:
    -----------
    data_dir : str
        Directory containing raw spatial omics data files.
    processed_dir : str
        Directory to save/load preprocessed graphs.
    batch_size_per_device : int
        Batch size for dataloaders.
    num_workers : int
        Number of workers for data loading.
    pin_memory : bool
        Whether to pin memory for DataLoader.
    min_cells : int
        Minimum number of cells required for preprocessing.
    min_genes : int
        Minimum number of genes required for preprocessing.
    graph_method : str
        Method for graph construction ("knn" or "pairwise").
    n_neighbors : int
        Number of neighbors for k-NN graph construction.
    graphs : list
        List of graph objects created from the dataset.
    dataset : SpatialOmicsDataset
        Dataset object containing graphs and metadata.
    """

    def __init__(
        self,
        data_dir: str = "data/domain/raw/",
        processed_dir: str = "data/domain/processed/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        min_cells: int = 3,
        min_genes: int = 3,
        n_pca_components = 50,
        graph_method: str = "knn",
        n_neighbors: int = 10,
        redo_preprocess: bool = False,
    ) -> None:
        """
        Initialize the SpatialOmicsDataModule.

        Parameters:
        ----------
        data_dir : str, optional
            Directory containing raw spatial omics data files. Default is "data/domain/raw/".
        processed_dir : str, optional
            Directory to save/load preprocessed graphs. Default is "data/domain/processed/".
        batch_size : int, optional
            Batch size for dataloaders. Default is 1.
        num_workers : int, optional
            Number of workers for data loading. Default is 0.
        pin_memory : bool, optional
            Whether to pin memory for DataLoader. Default is False.
        min_cells : int, optional
            Minimum number of cells required for preprocessing. Default is 3.
        min_genes : int, optional
            Minimum number of genes required for preprocessing. Default is 3.
        graph_method : str, optional
            Method for graph construction ("knn" or "pairwise"). Default is "knn".
        n_neighbors : int, optional
            Number of neighbors for k-NN graph construction. Default is 10.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = batch_size

        self.graphs = None
        self.dataset = None

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
            raise FileNotFoundError(f"Data directory {self.hparams.data_dir} does not exist. Please download the data.")
        else:
            log.info(f"Data directory {self.hparams.data_dir} exists. Proceeding with preprocessing or loading the data.")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load and preprocess spatial omics data.

        This method loads preprocessed graphs from disk if they exist. If not, it preprocesses
        the raw data, constructs graphs, and saves the preprocessed graphs to disk. This method 
        is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, 
        and `trainer.predict()`,

        Parameters:
        ----------
        stage : str, optional
            The stage of the setup process (e.g., "fit", "test"). Default is None.
        """
        # divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and process datasets only if not loaded already
        if not self.graphs and not self.dataset:
            processed_file = os.path.join(self.hparams.processed_dir, "dataset.pt")

            if os.path.exists(processed_file) and not self.hparams.redo_preprocess:
                # load preprocessed graphs and not redo preprocessing
                self.dataset = torch.load(processed_file, weights_only=False)
                log.info(f"Loaded preprocessed graphs from {os.path.join(self.hparams.processed_dir, processed_file)}.")
                # append each graph to self.graphs
                self.graphs = [self.dataset[i] for i in range(len(self.dataset))]
            else:
                # preprocess data and save to disk
                log.info(f"Preprocessing data from {self.hparams.data_dir} and saving to {self.hparams.processed_dir}.")
                # read samples into dict
                file_paths = [os.path.join(self.hparams.data_dir, f) for f in os.listdir(self.hparams.data_dir) if f.endswith(".h5ad")]
                samples = read_samples_into_dict(file_paths)

                # preprocess samples and create graphs
                self.graphs = []
                for sample_name, adata in samples.items():
                    if "X_pca" in adata.obsm:
                        log.info(f"Sample {sample_name} is already preprocessed. Skipping preprocessing.")
                    else:
                        preprocess_sample(adata, min_cells=self.hparams.min_cells, min_genes=self.hparams.min_genes, n_pca_components=self.hparams.n_pca_components)
                    graph = create_graph(adata, sample_name=sample_name, method=self.hparams.graph_method, n_neighbors=self.hparams.n_neighbors)
                    self.graphs.append(graph)
                    save_sample(adata, graph, self.hparams.processed_dir, sample_name)

                # save preprocessed graphs
                self.dataset = SpatialOmicsDataset({name: graph for name, graph in zip(samples.keys(), self.graphs)})
                torch.save(self.dataset, os.path.join(self.hparams.processed_dir, "dataset.pt"))
                log.info(f"Saved preprocessed graphs to {processed_file}. Finished preprocessing.")

    def train_dataloader(self) -> DataLoader:
        """
        Return the train dataloader.

        Returns:
        -------
        DataLoader
            A PyTorch Geometric DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
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
            dataset=self.dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
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
            dataset=self.dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
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