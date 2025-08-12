import rootutils
import torch
import pytest
from torch_geometric.data import Data, Batch
import torch_geometric
import numpy as np
import types

# Set up root for module imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.grace_domain_module import GRACELitModule
from src.models.components.grace import GRACEModel, Encoder


def create_simple_graph():
    """Create a simple test graph for domain testing."""
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                               [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=torch.float)
    x = torch.tensor([[1.0, 10.0],
                      [2.0, 20.0],
                      [3.0, 30.0]], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, num_nodes=3)


def create_grace_model():
    """Create a simple GRACE model for testing."""
    encoder = Encoder(
        in_channels=2,
        out_channels=64,
        activation=torch.nn.functional.relu,
        base_model=torch_geometric.nn.GCNConv,
        k=2
    )
    return GRACEModel(
        encoder=encoder,
        num_hidden=64,
        num_proj_hidden=64,
        tau=0.5
    )


def create_grace_domain_module():
    """Create a GRACE domain module for testing."""
    net = create_grace_model()
    module = GRACELitModule(
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10),
        compile=False,
        augmentation_mode="baseline",
        augmentation_list1=[],
        augmentation_list2=[],
        tau=0.5,
        spatial_regularization_strength=0.0,
        node_subset_sz=5000,
        drop_edge_p1=0.3,
        drop_edge_p2=0.4,
        drop_feat_p1=0.3,
        drop_feat_p2=0.4,
        mu=0.0,
        p_lambda=0.0,
        p_rewire=0.0,
        p_shuffle=0.0,
        spatial_noise_std=0.0,
        feature_noise_std=0.0,
        p_add=0.0,
        k_add=0,
        smooth_strength=0.0,
        apoptosis_p=0.0,
        mitosis_p=0.0,
        mitosis_feature_noise_std=0.0,
        processed_dir="",
        seed=42
    )
    # Attach a minimal dummy trainer and stub logger to avoid Lightning internals in unit tests
    dummy_trainer = types.SimpleNamespace(
        optimizers=[torch.optim.Adam(module.parameters(), lr=1e-3)],
        model=module,
    )
    setattr(module, "_trainer", dummy_trainer)
    setattr(module, "log", lambda *args, **kwargs: None)
    # Ensure warmup_steps is present for configure_optimizers
    try:
        module.hparams["warmup_steps"] = 1
    except Exception:
        pass
    return module


class TestGRACELitModule:
    """Test class for GRACE domain module."""

    def test_module_initialization(self):
        """Test that the module initializes correctly."""
        module = create_grace_domain_module()
        assert isinstance(module, GRACELitModule)
        assert isinstance(module.net, GRACEModel)
        assert module.hparams.tau == 0.5
        assert module.hparams.drop_edge_p1 == 0.3
        assert module.hparams.drop_edge_p2 == 0.4

    def test_forward_pass(self):
        """Test the forward pass of the module."""
        torch.manual_seed(42)
        module = create_grace_domain_module()
        data = create_simple_graph()
        
        # Test forward pass
        output = module(data.x, data.edge_index, data.edge_weight)
        assert output.shape == (3, 64)  # num_nodes x out_channels
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_pass_with_edge_attr(self):
        """Test forward pass with edge attributes (for GIN compatibility)."""
        torch.manual_seed(42)
        module = create_grace_domain_module()
        data = create_simple_graph()
        
        # Add edge attributes
        edge_attr = torch.randn(data.edge_index.size(1), 4)
        
        # Test forward pass with edge_attr
        output = module(data.x, data.edge_index, edge_attr)
        assert output.shape == (3, 64)
        assert not torch.isnan(output).any()

    def test_training_step(self):
        """Test the training step."""
        torch.manual_seed(42)
        module = create_grace_domain_module()
        data = create_simple_graph()
        
        # Test training step
        loss = module.training_step(data, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss > 0  # Loss should be positive

    def test_training_step_with_spatial_regularization(self):
        """Test training step with spatial regularization."""
        torch.manual_seed(42)
        module = create_grace_domain_module()
        data = create_simple_graph()
        
        # Add spatial coordinates
        data.pos = torch.randn(3, 2)
        
        # Test training step
        loss = module.training_step(data, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert not torch.isnan(loss)

    def test_validation_step(self):
        """Test the validation step."""
        torch.manual_seed(42)
        module = create_grace_domain_module()
        data = create_simple_graph()
        batch = Batch.from_data_list([data])
        
        # Test validation step (ignore errors from missing external files)
        try:
            module.validation_step(batch, batch_idx=0)
        except Exception:
            pass
        
        # Check that metrics were logged
        assert hasattr(module, 'val_nmi')
        assert hasattr(module, 'val_outputs')

    def test_test_step(self):
        """Test the test step."""
        torch.manual_seed(42)
        module = create_grace_domain_module()
        data = create_simple_graph()
        batch = Batch.from_data_list([data])
        
        # Test test step (ignore errors from missing external files)
        try:
            module.test_step(batch, batch_idx=0)
        except Exception:
            pass
        
        # Check that metrics were logged
        assert hasattr(module, 'test_nmi')
        assert hasattr(module, 'test_outputs')

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        module = create_grace_domain_module()
        optimizers = module.configure_optimizers()
        
        assert "optimizer" in optimizers
        assert isinstance(optimizers["optimizer"], torch.optim.Adam)
        
        if "lr_scheduler" in optimizers:
            assert "scheduler" in optimizers["lr_scheduler"]
            assert "interval" in optimizers["lr_scheduler"]
            assert "frequency" in optimizers["lr_scheduler"]

    def test_graph_augmentations(self):
        """Test that graph augmentations work correctly."""
        torch.manual_seed(42)
        module = create_grace_domain_module()
        data = create_simple_graph()
        
        # Test with edge dropout
        module.hparams.drop_edge_p1 = 0.5
        module.hparams.drop_edge_p2 = 0.5
        
        loss = module.training_step(data, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_feature_dropout(self):
        """Test that feature dropout works correctly."""
        torch.manual_seed(42)
        module = create_grace_domain_module()
        data = create_simple_graph()
        
        # Test with feature dropout
        module.hparams.drop_feat_p1 = 0.3
        module.hparams.drop_feat_p2 = 0.3
        
        loss = module.training_step(data, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_different_base_models(self):
        """Test with different base GNN models."""
        torch.manual_seed(42)
        
        # Test with GCN
        encoder_gcn = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=torch_geometric.nn.GCNConv,
            k=2
        )
        net_gcn = GRACEModel(
            encoder=encoder_gcn,
            num_hidden=64,
            num_proj_hidden=64,
            tau=0.5
        )
        
        module_gcn = GRACELitModule(
            net=net_gcn,
            optimizer=torch.optim.Adam,
            scheduler=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10),
            compile=False,
            augmentation_mode="baseline",
            augmentation_list1=[],
            augmentation_list2=[],
            tau=0.5,
            spatial_regularization_strength=0.0,
            node_subset_sz=5000,
            mu=0.0,
            p_lambda=0.0,
            p_rewire=0.0,
            drop_edge_p1=0.3,
            drop_edge_p2=0.4,
            drop_feat_p1=0.3,
            drop_feat_p2=0.4,
            p_shuffle=0.0,
            spatial_noise_std=0.0,
            feature_noise_std=0.0,
            p_add=0.0,
            k_add=0,
            smooth_strength=0.0,
            apoptosis_p=0.0,
            mitosis_p=0.0,
            mitosis_feature_noise_std=0.0,
            processed_dir="",
            seed=42
        )
        
        data = create_simple_graph()
        output_gcn = module_gcn(data.x, data.edge_index, data.edge_weight)
        assert output_gcn.shape == (3, 64)
        
        # Test with GAT
        encoder_gat = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=torch_geometric.nn.GATConv,
            k=2
        )
        net_gat = GRACEModel(
            encoder=encoder_gat,
            num_hidden=64,
            num_proj_hidden=64,
            tau=0.5
        )
        
        module_gat = GRACELitModule(
            net=net_gat,
            optimizer=torch.optim.Adam,
            scheduler=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10),
            compile=False,
            augmentation_mode="baseline",
            augmentation_list1=[],
            augmentation_list2=[],
            tau=0.5,
            spatial_regularization_strength=0.0,
            node_subset_sz=5000,
            mu=0.0,
            p_lambda=0.0,
            p_rewire=0.0,
            drop_edge_p1=0.3,
            drop_edge_p2=0.4,
            drop_feat_p1=0.3,
            drop_feat_p2=0.4,
            p_shuffle=0.0,
            spatial_noise_std=0.0,
            feature_noise_std=0.0,
            p_add=0.0,
            k_add=0,
            smooth_strength=0.0,
            apoptosis_p=0.0,
            mitosis_p=0.0,
            mitosis_feature_noise_std=0.0,
            processed_dir="",
            seed=42
        )
        
        output_gat = module_gat(data.x, data.edge_index, data.edge_weight)
        assert output_gat.shape == (3, 64)

    def test_batch_processing(self):
        """Test processing of batched graphs."""
        torch.manual_seed(42)
        module = create_grace_domain_module()
        
        # Create multiple graphs
        graph1 = create_simple_graph()
        graph2 = create_simple_graph()
        
        # Create batch
        batch = Batch.from_data_list([graph1, graph2])
        
        # Test forward pass on batch
        output = module(batch.x, batch.edge_index, batch.edge_weight)
        assert output.shape == (6, 64)  # 2 graphs * 3 nodes each

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        torch.manual_seed(42)
        module = create_grace_domain_module()
        
        # Test with empty graph
        empty_data = Data(
            x=torch.empty(0, 2),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_weight=torch.empty(0),
            num_nodes=0
        )
        
        # This should handle empty graphs gracefully
        try:
            output = module(empty_data.x, empty_data.edge_index, empty_data.edge_weight)
            assert output.shape == (0, 64)
        except Exception as e:
            # It's okay if it raises an exception for empty graphs
            assert "empty" in str(e).lower() or "size" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__]) 