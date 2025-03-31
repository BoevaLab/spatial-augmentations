import torch
import scanpy as sc
from torch_geometric.data import Data
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.spatial_omics_datamodule import SpatialOmicsDataModule


def test_spatial_omics_datamodule():

    # initialize the datamodule
    datamodule = SpatialOmicsDataModule(
        data_dir="data/domain/raw/",
        processed_dir="data/domain/processed/",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        min_cells=3,
        min_genes=3,
        graph_method="knn",
        n_neighbors=10,
        redo_preprocess=True,
    )

    assert datamodule.hparams.data_dir == "data/domain/raw/"
    assert datamodule.hparams.min_cells == 3

    datamodule.prepare_data()
    datamodule.setup()

    print(datamodule.dataset)
    print(len(datamodule.graphs))

    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    val_loader = datamodule.val_dataloader()

    for batch in train_loader:
        print(batch)
    for batch in test_loader:
        print(batch)
    for batch in val_loader:
        print(batch)
    
    print("Test passed!")

if __name__ == "__main__":
    test_spatial_omics_datamodule()