import rootutils
import torch
import pytest
from torch_geometric.data import Data, Batch
import torch_geometric
import numpy as np
import tempfile
import os
import types

# Set up root for module imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.grace_phenotype_module import GRACEPhenotypeLitModule
from src.models.components.grace import GRACEModel, Encoder
from src.models.components.grace_pred import GRACE_pred


def create_simple_graph():
    """Create a simple test graph for phenotype testing."""
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                               [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=torch.float)
    x = torch.tensor([[1.0, 10.0],
                      [2.0, 20.0],
                      [3.0, 30.0]], dtype=torch.float)
    # Use edge_attr for compatibility with augmentations
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight.view(-1, 1), num_nodes=3)


def create_phenotype_graph():
    """Create a test graph with phenotype labels."""
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                               [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=torch.float)
    x = torch.tensor([[1.0, 10.0],
                      [2.0, 20.0],
                      [3.0, 30.0]], dtype=torch.float)
    
    # Add phenotype labels
    y = torch.tensor([1.0], dtype=torch.float)  # Binary classification
    w = torch.tensor([1.0], dtype=torch.float)  # Sample weights
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, 
                y=y, w=w, num_nodes=3, region_id="test_region")


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


def create_grace_pred_model():
    """Create a simple GRACE_pred model for testing."""
    return GRACE_pred(
        num_layer=2,
        in_channels=2,
        emb_dim=64,
        num_additional_feat=0,
        num_graph_tasks=1,
        node_embedding_output="last",
        drop_ratio=0.1,
        graph_pooling="max",
        gnn_type="gcn",
        activation=torch.nn.functional.relu
    )


def create_grace_phenotype_module(mode="pretraining", net=None):
    """Create a GRACE phenotype module for testing."""
    if net is None:
        if mode == "pretraining":
            net = create_grace_model()
        else:
            net = create_grace_pred_model()
    
    module = GRACEPhenotypeLitModule(
        mode=mode,
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10),
        test_thresh=0.5,
        compile=False,
        augmentation_mode="baseline",
        augmentation_list1=[],
        augmentation_list2=[],
        tau=0.5,
        drop_edge_p1=0.3,
        drop_edge_p2=0.4,
        drop_feat_p1=0.3,
        drop_feat_p2=0.4,
        mu=0.0,
        p_lambda=0.0,
        p_rewire=0.0,
        feature_noise_std=0.0,
        p_add=0.0,
        k_add=0,
        p_shuffle=0.0,
        apoptosis_p=0.0,
        mitosis_p=0.0,
        shift_p=0.0,
        shift_map={},
        seed=42,
        ckpt_file=None
    )
    # Attach a minimal dummy trainer and stub logger to avoid Lightning internals in unit tests
    setattr(module, "_trainer", types.SimpleNamespace(optimizers=[torch.optim.Adam(module.parameters(), lr=1e-3)], model=module))
    setattr(module, "log", lambda *args, **kwargs: None)
    try:
        module.hparams["warmup_steps"] = 1
    except Exception:
        pass
    return module


class TestGRACEPhenotypeModule:
    """Test class for GRACE phenotype module."""

    def test_module_initialization_pretraining(self):
        """Test that the module initializes correctly in pretraining mode."""
        module = create_grace_phenotype_module(mode="pretraining")
        assert isinstance(module, GRACEPhenotypeLitModule)
        assert isinstance(module.net, GRACEModel)
        assert module.hparams.mode == "pretraining"
        assert module.hparams.tau == 0.5

    def test_module_initialization_finetuning(self):
        """Test that the module initializes correctly in finetuning mode."""
        module = create_grace_phenotype_module(mode="finetuning")
        assert isinstance(module, GRACEPhenotypeLitModule)
        assert isinstance(module.net, GRACE_pred)
        assert module.hparams.mode == "finetuning"

    def test_forward_pass_pretraining(self):
        """Test the forward pass in pretraining mode."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="pretraining")
        data = create_simple_graph()
        
        # Test forward pass
        output = module(data.x, data.edge_index, data.edge_weight)
        assert output.shape == (3, 64)  # num_nodes x out_channels
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_pass_finetuning(self):
        """Test the forward pass in finetuning mode."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="finetuning")
        data = create_phenotype_graph()
        
        # Test forward pass
        output = module(data)
        assert output.shape == (1,)  # Single prediction for graph
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_gnn_pred(self):
        """Test the forward_gnn_pred method."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="finetuning")
        data = create_phenotype_graph()
        
        # Test forward_gnn_pred
        output = module.forward_gnn_pred(data)
        assert isinstance(output, list)
        assert len(output) == 1  # Single prediction
        assert not torch.isnan(output[0]).any()

    def test_training_step_pretraining(self):
        """Test the training step in pretraining mode."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="pretraining")
        data = create_simple_graph()
        
        # Test training step
        loss = module.training_step(data, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss > 0  # Loss should be positive

    def test_training_step_finetuning(self):
        """Test the training step in finetuning mode."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="finetuning")
        data = create_phenotype_graph()
        
        # Test training step
        loss = module.training_step(data, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_validation_step(self):
        """Test the validation step."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="finetuning")
        data = create_phenotype_graph()
        
        # Test validation step
        module.validation_step(Batch.from_data_list([data]), batch_idx=0)
        
        # Check that metrics were logged
        assert hasattr(module, 'val_loss')
        assert module.val_loss is not None

    def test_test_step(self):
        """Test the test step."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="finetuning")
        data = create_phenotype_graph()
        
        # Test test step
        module.test_step(Batch.from_data_list([data]), batch_idx=0)
        
        # Check that outputs were stored
        assert hasattr(module, 'test_outputs')
        assert len(module.test_outputs) > 0

    def test_calculate_metrics(self):
        """Test the calculate_metrics method."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="finetuning")
        
        # Create test predictions and labels
        logits = torch.tensor([0.8, 0.2, 0.9, 0.1])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        # Test metrics calculation
        metrics = module.calculate_metrics(logits, labels, test=False)
        
        # Check that all expected metrics are present
        expected_metrics = ['auroc', 'accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])

    def test_binary_cross_entropy_loss(self):
        """Test the binary cross entropy loss function."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="finetuning")
        
        # Create test predictions and labels
        y_pred = torch.tensor([0.8, 0.2, 0.9, 0.1])
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
        
        # Test loss calculation
        loss = module.binary_cross_entropy_loss(y_pred, y_true, weight)
        assert isinstance(loss, torch.Tensor)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_model_loading_pretrained_to_finetuning(self):
        """Test loading pretrained model for finetuning."""
        torch.manual_seed(42)
        
        # Create pretraining module and train it briefly
        pretrain_module = create_grace_phenotype_module(mode="pretraining")
        data = create_simple_graph()
        
        # Save a checkpoint
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Save the pretrained model
            torch.save({
                'state_dict': pretrain_module.state_dict()
            }, checkpoint_path)
            
            # Create finetuning module
            finetune_module = create_grace_phenotype_module(mode="finetuning")
            
            # Load pretrained weights
            finetune_module.load_pretrained_model(checkpoint_path)
            
            # Check that the encoder weights were loaded
            if isinstance(finetune_module.net, GRACE_pred):
                assert hasattr(finetune_module.net, 'encoder')
                # The encoder should have the same weights as the pretrained model
                pretrain_encoder_state = pretrain_module.net.encoder.state_dict()
                finetune_encoder_state = finetune_module.net.encoder.state_dict()
                
                for key in pretrain_encoder_state:
                    assert key in finetune_encoder_state
                    assert torch.allclose(
                        pretrain_encoder_state[key], 
                        finetune_encoder_state[key]
                    )
            
        finally:
            # Clean up
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)

    def test_two_phase_training_workflow(self):
        """Test the complete two-phase training workflow."""
        torch.manual_seed(42)
        
        # Phase 1: Pretraining
        pretrain_module = create_grace_phenotype_module(mode="pretraining")
        data = create_simple_graph()
        
        # Simulate pretraining
        pretrain_loss = pretrain_module.training_step(data, batch_idx=0)
        assert isinstance(pretrain_loss, torch.Tensor)
        assert pretrain_loss > 0
        
        # Phase 2: Finetuning
        finetune_module = create_grace_phenotype_module(mode="finetuning")
        phenotype_data = create_phenotype_graph()
        
        # Simulate finetuning
        finetune_loss = finetune_module.training_step(phenotype_data, batch_idx=0)
        assert isinstance(finetune_loss, torch.Tensor)
        assert finetune_loss > 0

    def test_different_gnn_types(self):
        """Test with different GNN types in GRACE_pred."""
        torch.manual_seed(42)
        
        # Test with GCN
        gcn_pred = GRACE_pred(
            num_layer=2,
            in_channels=2,
            emb_dim=64,
            num_additional_feat=0,
            num_graph_tasks=1,
            node_embedding_output="last",
            drop_ratio=0.1,
            graph_pooling="max",
            gnn_type="gcn",
            activation=torch.nn.functional.relu
        )
        
        module_gcn = create_grace_phenotype_module(mode="finetuning", net=gcn_pred)
        data = create_phenotype_graph()
        output_gcn = module_gcn(data.x, data.edge_index, data.edge_weight)
        assert output_gcn.shape == (1,)
        
        # Test with GIN
        gin_pred = GRACE_pred(
            num_layer=2,
            in_channels=2,
            emb_dim=64,
            num_additional_feat=0,
            num_graph_tasks=1,
            node_embedding_output="last",
            drop_ratio=0.1,
            graph_pooling="max",
            gnn_type="gin",
            activation=torch.nn.functional.relu
        )
        
        module_gin = create_grace_phenotype_module(mode="finetuning", net=gin_pred)
        output_gin = module_gin(data.x, data.edge_index, data.edge_weight)
        assert output_gin.shape == (1,)

    def test_edge_attr_compatibility(self):
        """Test compatibility with edge attributes for GIN."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="pretraining")
        data = create_simple_graph()
        
        # Add edge attributes
        edge_attr = torch.randn(data.edge_index.size(1), 4)
        
        # Test forward pass with edge_attr
        output = module(data.x, data.edge_index, edge_attr)
        assert output.shape == (3, 64)
        assert not torch.isnan(output).any()

    def test_batch_processing(self):
        """Test processing of batched graphs."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="finetuning")
        
        # Create multiple graphs
        graph1 = create_phenotype_graph()
        graph2 = create_phenotype_graph()
        
        # Test individual processing
        output1 = module(graph1.x, graph1.edge_index, graph1.edge_weight)
        output2 = module(graph2.x, graph2.edge_index, graph2.edge_weight)
        
        assert output1.shape == (1,)
        assert output2.shape == (1,)

    def test_metrics_aggregation(self):
        """Test metrics aggregation in validation and test steps."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="finetuning")
        
        # Create multiple test samples
        data1 = create_phenotype_graph()
        data2 = create_phenotype_graph()
        
        # Test validation step with multiple samples (invoke per-sample to collect separate outputs)
        module.validation_step(Batch.from_data_list([data1]), batch_idx=0)
        module.validation_step(Batch.from_data_list([data2]), batch_idx=1)
        assert hasattr(module, 'val_loss')
        
        # Test test step with multiple samples (invoke per-sample to collect separate outputs)
        module.test_step(Batch.from_data_list([data1]), batch_idx=0)
        module.test_step(Batch.from_data_list([data2]), batch_idx=1)
        assert hasattr(module, 'test_outputs')
        assert len(module.test_outputs) == 2

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        torch.manual_seed(42)
        module = create_grace_phenotype_module(mode="finetuning")
        
        # Test with empty graph
        empty_data = Data(
            x=torch.empty(0, 2),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_weight=torch.empty(0),
            y=torch.tensor([0.0]),
            w=torch.tensor([1.0]),
            num_nodes=0,
            region_id="empty_region"
        )
        
        # This should handle empty graphs gracefully
        try:
            output = module(empty_data.x, empty_data.edge_index, empty_data.edge_weight)
            assert output.shape == (1,)  # Still produces a prediction
        except Exception as e:
            # It's okay if it raises an exception for empty graphs
            assert "empty" in str(e).lower() or "size" in str(e).lower()

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        module = create_grace_phenotype_module(mode="finetuning")
        optimizers = module.configure_optimizers()
        
        assert "optimizer" in optimizers
        assert isinstance(optimizers["optimizer"], torch.optim.Adam)
        
        if "lr_scheduler" in optimizers:
            assert "scheduler" in optimizers["lr_scheduler"]
            assert "interval" in optimizers["lr_scheduler"]
            assert "frequency" in optimizers["lr_scheduler"]


if __name__ == "__main__":
    pytest.main([__file__]) 