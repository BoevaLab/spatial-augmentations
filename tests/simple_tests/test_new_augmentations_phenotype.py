import rootutils
import torch
import pytest
from torch_geometric.data import Data
import torch_geometric

# Set up root for module imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import phenotype augmentations
from src.utils.graph_augmentations_phenotype import (
    Apoptosis,
    Mitosis,
    PhenotypeShift,
    AddEdgesByCellType,
)

# Helper to create a simple test graph

def create_simple_graph():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                               [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    edge_attr = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=torch.float)
    # x: first column phenotype labels, second arbitrary feature
    x = torch.tensor([[0.0, 10.0],
                      [1.0, 20.0],
                      [2.0, 30.0]], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)

# ------------------ Apoptosis Tests ------------------

def test_apoptosis_no_drop():
    torch.manual_seed(0)
    data = create_simple_graph()
    aug = Apoptosis(apoptosis_p=0.0)
    out = aug(data.clone())
    assert out.num_nodes == 3
    ei_ref, ew_ref = torch_geometric.utils.coalesce(
        data.edge_index, data.edge_attr, num_nodes=3
    )
    ei_out, ew_out = torch_geometric.utils.coalesce(
        out.edge_index, out.edge_attr, num_nodes=3
    )
    assert torch.equal(ei_ref, ei_out)
    assert torch.allclose(ew_ref, ew_out)


def test_apoptosis_all_drop():
    torch.manual_seed(0)
    data = create_simple_graph()
    aug = Apoptosis(apoptosis_p=1.0)
    out = aug(data.clone())
    assert out.num_nodes == 0
    assert out.edge_index.numel() == 0
    assert hasattr(out, 'x') and out.x.numel() == 0

# ------------------ Mitosis Tests ------------------

def test_mitosis_single_node():
    torch.manual_seed(0)
    x = torch.tensor([[5.0, 50.0]], dtype=torch.float)
    data = Data(x=x.clone(), edge_index=torch.empty((2,0), dtype=torch.long), num_nodes=1)
    aug = Mitosis(mitosis_p=1.0)
    out = aug(data.clone())
    assert out.num_nodes == 2
    # should have exactly two directed edges between 0 and 1
    sorted_edges = set(tuple(e) for e in out.edge_index.t().tolist())
    assert {(0,1),(1,0)}.issubset(sorted_edges)
    # features identical
    assert torch.allclose(out.x[0], out.x[1])


def test_mitosis_edge_inheritance():
    torch.manual_seed(0)
    edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    x = torch.tensor([[1.0],[2.0]], dtype=torch.float)
    edge_attr = torch.tensor([7.0,7.0], dtype=torch.float)
    data = Data(x=x.clone(), edge_index=edge_index, edge_attr=edge_attr, num_nodes=2)
    aug = Mitosis(mitosis_p=1.0)
    out = aug(data.clone())
    assert out.num_nodes == 4
    # node 0 neighbors include original 1, its clone 2 and clone of neighbor 3
    nbrs0 = set(out.edge_index[1, out.edge_index[0]==0].tolist())
    assert {1,2,3}.issubset(nbrs0)
    # clone of 0 is node 2: neighbors include 0 and 1
    nbrs2 = set(out.edge_index[1, out.edge_index[0]==2].tolist())
    assert {0,1}.issubset(nbrs2)

# --------------- PhenotypeShift Tests ---------------

def test_phenotype_shift_all():
    torch.manual_seed(0)
    data = create_simple_graph()
    # map 0->2, 1->0; leave 2 unchanged
    shift_map = {0:[2],1:[0]}
    aug = PhenotypeShift(shift_p=1.0, shift_map=shift_map, cell_type_feat=0)
    out = aug(data.clone())
    # all nodes attempted; new labels:
    assert out.x[0,0] == 2.0  # was 0 -> 2
    assert out.x[1,0] == 0.0  # was 1 -> 0
    assert out.x[2,0] == 2.0  # was 2, unchanged


def test_phenotype_shift_partial():
    torch.manual_seed(1)
    data = create_simple_graph()
    shift_map = {0:[1,2]}
    aug = PhenotypeShift(shift_p=0.5, shift_map=shift_map, cell_type_feat=0)
    out = aug(data.clone())
    # exactly half the nodes (on average) will shift; check that any 0-> either 1 or 2 when shifted
    for orig, new in zip(data.x[:,0], out.x[:,0]):
        if new != orig:
            assert new in torch.tensor([1.0,2.0])

# Run with pytest

if __name__ == '__main__':
    pytest.main([__file__])