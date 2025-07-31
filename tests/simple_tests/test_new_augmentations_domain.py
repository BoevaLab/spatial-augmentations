import rootutils
import torch
import pytest
from torch_geometric.data import Data
import torch_geometric
import numpy as np

# Set up root for module imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# DataModule and augmentation factory
from src.data.spatial_omics_datamodule import SpatialOmicsDataModule
from src.utils.graph_augmentations_domain import (
    get_graph_augmentation,
    Apoptosis,
    Mitosis,
    SmoothFeatures,
)

# Helper to create a simple test graph
def create_simple_graph():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                               [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=torch.float)
    x = torch.tensor([[1.0, 10.0],
                      [2.0, 20.0],
                      [3.0, 30.0]], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, num_nodes=3)

# Random seed will be set in each test for reproducibility

# -- Apoptosis Tests --
def test_apoptosis_no_drop():
    torch.manual_seed(42)
    data = create_simple_graph()
    aug = Apoptosis(apoptosis_p=0.0)
    out = aug(data.clone())
    assert out.num_nodes == 3
    ei_ref, ew_ref = torch_geometric.utils.coalesce(data.edge_index, data.edge_weight, num_nodes=3)
    ei_out, ew_out = torch_geometric.utils.coalesce(out.edge_index, out.edge_weight, num_nodes=3)
    assert torch.equal(ei_ref, ei_out)
    assert torch.allclose(ew_ref, ew_out)


def test_apoptosis_all_drop():
    torch.manual_seed(42)
    data = create_simple_graph()
    aug = Apoptosis(apoptosis_p=1.0)
    out = aug(data.clone())
    assert out.num_nodes == 0
    assert out.edge_index.numel() == 0
    assert out.x is not None and out.x.numel() == 0

# -- Mitosis Tests --
def test_mitosis_single_node():
    torch.manual_seed(42)
    x = torch.tensor([[5.0, 50.0]], dtype=torch.float)
    data = Data(x=x.clone(), edge_index=torch.empty((2,0), dtype=torch.long), num_nodes=1)
    aug = Mitosis(mitosis_p=1.0, mitosis_feature_noise_std=0.0)
    out = aug(data.clone())
    assert out.num_nodes == 2
    assert out.edge_index.size(1) == 2
    assert torch.allclose(out.x[0], out.x[1])


def test_mitosis_edge_inheritance():
    # Set random seed for reproducible mitosis selection
    torch.manual_seed(42)
    edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    x = torch.tensor([[1.0],[2.0]], dtype=torch.float)
    edge_weight = torch.tensor([7.0,7.0])
    data = Data(x=x.clone(), edge_index=edge_index, edge_weight=edge_weight, num_nodes=2)
    aug = Mitosis(mitosis_p=1.0, mitosis_feature_noise_std=0.0)
    out = aug(data.clone())
    assert out.num_nodes == 4
    neighbors_0 = set(out.edge_index[1, out.edge_index[0]==0].tolist())
    neighbors_2 = set(out.edge_index[1, out.edge_index[0]==2].tolist())
    print(f"All edges: {out.edge_index.t().tolist()}")
    assert {1, 2, 3}.issubset(neighbors_0)
    assert {0, 1}.issubset(neighbors_2)

# -- SmoothFeatures Tests --
def test_smooth_features_strength_one():
    edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    x = torch.tensor([[1.0],[3.0]], dtype=torch.float)
    data = Data(x=x.clone(), edge_index=edge_index, num_nodes=2)
    aug = SmoothFeatures(smooth_strength=1.0)
    out = aug(data.clone())
    assert torch.allclose(out.x[0], torch.tensor([3.0]))
    assert torch.allclose(out.x[1], torch.tensor([1.0]))


def test_smooth_features_strength_half():
    edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    x = torch.tensor([[1.0],[3.0]], dtype=torch.float)
    data = Data(x=x.clone(), edge_index=edge_index, num_nodes=2)
    aug = SmoothFeatures(smooth_strength=0.5)
    out = aug(data.clone())
    assert torch.allclose(out.x, torch.tensor([[2.0],[2.0]]))

def test_smooth_features_strength_alpha():
    edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    x = torch.tensor([[1.0],[3.0]], dtype=torch.float)
    data = Data(x=x.clone(), edge_index=edge_index, num_nodes=2)
    aug = SmoothFeatures(smooth_strength=0.1)
    out = aug(data.clone())
    assert torch.allclose(out.x, torch.tensor([[1.2],[2.8]]))