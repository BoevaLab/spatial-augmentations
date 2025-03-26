from typing import Any, Dict, List, Tuple
import os
import hydra
import torch
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger
from src.utils.preprocess_helpers import (
    read_samples_into_dict,
    save_sample,
    preprocess_sample,
    create_graph,
    SpatialOmicsDataset
)

log = RankedLogger(__name__, rank_zero_only=True)


def preprocess(cfg: DictConfig) -> None:
    """
    Preprocesses spatial omics data and creates graph representations.

    This function reads raw `.h5ad` files, preprocesses the data, and generates graph representations
    for each sample. The processed data and graphs are saved to the specified output directory.

    :param cfg: A DictConfig configuration composed by Hydra.
    """
    output_dir = cfg.data_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # read samples from input file paths into dicitonary
    file_paths = cfg.input_files
    log.info(f"Found {len(file_paths)} input files.")
    samples = read_samples_into_dict(file_paths)

    # read parameters from config
    min_cells = cfg.min_cells
    min_genes = cfg.min_genes
    n_neighbors = cfg.n_neighbors
    graph_method = cfg.graph_method
    graphs = []

    # preprocess each sample and create graphs
    for sample_name, adata in samples.items():
        if "X_pca" in adata.obsm:
            log.info(f"Sample {sample_name} is already preprocessed. Skipping preprocessing.")
        else:
            preprocess_sample(adata, min_cells=min_cells, min_genes=min_genes)
        graph = create_graph(adata, method=graph_method, n_neighbors=n_neighbors)
        graphs.append(graph)
        save_sample(adata, graph, output_dir, sample_name)

    # optionally save all graphs as a dataset
    if cfg.save_combined_dataset:
        log.info("Saving combined dataset...")
        dataset = SpatialOmicsDataset({name: graph for name, graph in zip(samples.keys(), graphs)})
        torch.save(dataset, os.path.join(output_dir, "domain_dataset.pt"))
        log.info(f"Combined dataset saved to {os.path.join(output_dir, 'domain_dataset.pt')}")

    log.info("Preprocessing complete!")


@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for preprocessing.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    preprocess(cfg)


if __name__ == "__main__":
    main()