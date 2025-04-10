import rootutils
import torch
from torch_geometric.data import Data

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.spatial_omics_datamodule import SpatialOmicsDataModule
from src.utils.graph_augmentations import get_graph_augmentation


def test_get_graph_augmentation():
    # mock augmentation parameters
    augmentation = "baseline"
    drop_edge_p = 0.1
    drop_feat_p = 0.1

    # Mock batch input (adjust based on your actual batch structure)
    # Initialize the SpatialOmicsDataModule
    data_module = SpatialOmicsDataModule(
        data_dir="data/domain/raw", processed_dir="data/domain/processed"
    )
    data_module.setup()

    # Get a batch from the test dataloader
    train_dataloader = data_module.train_dataloader()
    batch = next(iter(train_dataloader))

    print("Original batch:")
    print(batch)
    print(batch.sample_name)

    # Get the augmentation function
    transform = get_graph_augmentation(augmentation, drop_edge_p, drop_feat_p)

    # Apply the transformation
    result = transform(batch)

    # Print the result
    print("Result of get_graph_augmentation:")
    print(result)
    print(result.sample_name)


if __name__ == "__main__":
    test_get_graph_augmentation()
