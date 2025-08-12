import rootutils
import torch
import pytest
from torch_geometric.data import Data
import torch_geometric
import numpy as np

# Set up root for module imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.grace import GRACEModel, Encoder
from src.models.components.grace_pred import GRACE_pred
from src.models.components.gnn import GINConv, GCNConv, GATConv


def create_simple_graph():
    """Create a simple test graph."""
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                               [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    # Edge attributes: first column is edge type in {0,1} to satisfy embedding
    edge_attr = torch.randint(0, 2, (edge_index.size(1), 4), dtype=torch.long).float()
    x = torch.tensor([[1.0, 10.0],
                      [2.0, 20.0],
                      [3.0, 30.0]], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)


class TestGRACEModel:
    """Test class for GRACE model components."""

    def test_encoder_initialization(self):
        """Test encoder initialization with different base models."""
        # Test GCN
        encoder_gcn = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GCNConv,
            k=2
        )
        assert isinstance(encoder_gcn, Encoder)
        assert len(encoder_gcn.gnn.gnns) == 2
        
        # Test GAT
        encoder_gat = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GATConv,
            k=2
        )
        assert isinstance(encoder_gat, Encoder)
        assert len(encoder_gat.gnn.gnns) == 2
        
        # Test GIN
        encoder_gin = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GINConv,
            k=2
        )
        assert isinstance(encoder_gin, Encoder)
        assert len(encoder_gin.gnn.gnns) == 2

    def test_encoder_forward_gcn(self):
        """Test encoder forward pass with GCN."""
        torch.manual_seed(42)
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GCNConv,
            k=2
        )
        data = create_simple_graph()
        
        output = encoder(data.x, data.edge_index, data.edge_attr)
        assert output.shape == (3, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_encoder_forward_gin(self):
        """Test encoder forward pass with GIN (edge attributes)."""
        torch.manual_seed(42)
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GINConv,
            k=2
        )
        data = create_simple_graph()
        
        # Add edge attributes for GIN
        edge_attr = torch.randn(data.edge_index.size(1), 4)
        
        output = encoder(data.x, data.edge_index, edge_weight=edge_attr)
        assert output.shape == (3, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_encoder_forward_gat(self):
        """Test encoder forward pass with GAT."""
        torch.manual_seed(42)
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GATConv,
            k=2
        )
        data = create_simple_graph()
        
        output = encoder(data.x, data.edge_index, edge_attr=data.edge_attr)
        assert output.shape == (3, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_grace_model_initialization(self):
        """Test GRACE model initialization."""
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GCNConv,
            k=2
        )
        
        model = GRACEModel(
            encoder=encoder,
            num_hidden=64,
            num_proj_hidden=64,
            tau=0.5
        )
        
        assert isinstance(model, GRACEModel)
        assert isinstance(model.encoder, Encoder)
        assert model.tau == 0.5

    def test_grace_model_forward(self):
        """Test GRACE model forward pass."""
        torch.manual_seed(42)
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GCNConv,
            k=2
        )
        
        model = GRACEModel(
            encoder=encoder,
            num_hidden=64,
            num_proj_hidden=64,
            tau=0.5
        )
        
        data = create_simple_graph()
        output = model(data.x, data.edge_index, edge_attr=data.edge_attr)
        assert output.shape == (3, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_grace_model_forward_with_edge_attr(self):
        """Test GRACE model forward pass with edge attributes."""
        torch.manual_seed(42)
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GINConv,
            k=2
        )
        
        model = GRACEModel(
            encoder=encoder,
            num_hidden=64,
            num_proj_hidden=64,
            tau=0.5
        )
        
        data = create_simple_graph()
        # Use consistent edge_attr format expected by custom GIN/GCN layers
        output = model(data.x, data.edge_index, edge_attr=data.edge_attr)
        assert output.shape == (3, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestGRACEPred:
    """Test class for GRACE_pred model."""

    def test_grace_pred_initialization(self):
        """Test GRACE_pred initialization with different GNN types."""
        # Test GCN
        model_gcn = GRACE_pred(
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
        assert isinstance(model_gcn, GRACE_pred)
        assert hasattr(model_gcn, 'encoder')
        assert hasattr(model_gcn, 'graph_pred_module')
        
        # Test GIN
        model_gin = GRACE_pred(
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
        assert isinstance(model_gin, GRACE_pred)
        assert hasattr(model_gin, 'encoder')
        
        # Test GAT
        model_gat = GRACE_pred(
            num_layer=2,
            in_channels=2,
            emb_dim=64,
            num_additional_feat=0,
            num_graph_tasks=1,
            node_embedding_output="last",
            drop_ratio=0.1,
            graph_pooling="max",
            gnn_type="gat",
            activation=torch.nn.functional.relu
        )
        assert isinstance(model_gat, GRACE_pred)
        assert hasattr(model_gat, 'encoder')

    def test_grace_pred_forward_gcn(self):
        """Test GRACE_pred forward pass with GCN."""
        torch.manual_seed(42)
        model = GRACE_pred(
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
        
        data = create_simple_graph()
        output = model(data)
        
        assert output[0].shape == torch.Size([1, 1])  # Single graph prediction
        assert not torch.isnan(output[0]).any()
        assert not torch.isinf(output[0]).any()

    def test_grace_pred_forward_gin(self):
        """Test GRACE_pred forward pass with GIN."""
        torch.manual_seed(42)
        model = GRACE_pred(
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
        
        data = create_simple_graph()
        # Add edge attributes for GIN
        data.edge_attr = torch.randn(data.edge_index.size(1), 4)
        
        output = model(data)
        assert output[0].shape == torch.Size([1, 1])
        assert not torch.isnan(output[0]).any()
        assert not torch.isinf(output[0]).any()

    def test_grace_pred_different_pooling(self):
        """Test GRACE_pred with different pooling methods."""
        torch.manual_seed(42)
        
        for pooling in ["max", "mean", "sum"]:
            model = GRACE_pred(
                num_layer=2,
                in_channels=2,
                emb_dim=64,
                num_additional_feat=0,
                num_graph_tasks=1,
                node_embedding_output="last",
                drop_ratio=0.1,
                graph_pooling=pooling,
                gnn_type="gcn",
                activation=torch.nn.functional.relu
            )
            
            data = create_simple_graph()
            output = model(data)
            assert output[0].shape == torch.Size([1, 1])
            assert not torch.isnan(output[0]).any()

    def test_grace_pred_different_node_embedding_output(self):
        """Test GRACE_pred with different node embedding output modes."""
        torch.manual_seed(42)
        
        for output_mode in ["last", "concat", "max"]:
            model = GRACE_pred(
                num_layer=2,
                in_channels=2,
                emb_dim=64,
                num_additional_feat=0,
                num_graph_tasks=1,
                node_embedding_output=output_mode,
                drop_ratio=0.1,
                graph_pooling="max",
                gnn_type="gcn",
                activation=torch.nn.functional.relu
            )
            
            data = create_simple_graph()
            output = model(data)
            assert output[0].shape == torch.Size([1, 1])
            assert not torch.isnan(output[0]).any()

    def test_grace_pred_multiple_tasks(self):
        """Test GRACE_pred with multiple graph tasks."""
        torch.manual_seed(42)
        model = GRACE_pred(
            num_layer=2,
            in_channels=2,
            emb_dim=64,
            num_additional_feat=0,
            num_graph_tasks=3,  # Multiple tasks
            node_embedding_output="last",
            drop_ratio=0.1,
            graph_pooling="max",
            gnn_type="gcn",
            activation=torch.nn.functional.relu
        )
        
        data = create_simple_graph()
        output = model(data)
        assert output[0].shape == torch.Size([1, 3])  # 3 task predictions
        assert not torch.isnan(output[0]).any()

    def test_grace_pred_with_additional_features(self):
        """Test GRACE_pred with additional features."""
        torch.manual_seed(42)
        model = GRACE_pred(
            num_layer=2,
            in_channels=2,
            emb_dim=64,
            num_additional_feat=5,  # Additional features
            num_graph_tasks=1,
            node_embedding_output="last",
            drop_ratio=0.1,
            graph_pooling="max",
            gnn_type="gcn",
            activation=torch.nn.functional.relu
        )
        
        data = create_simple_graph()
        # Add graph-level additional features (num_graphs x F)
        data.additional_feat = torch.randn(1, 5)
        
        output = model(data)
        assert output[0].shape == torch.Size([1, 1])
        assert not torch.isnan(output[0]).any()

    def test_grace_pred_edge_cases(self):
        """Test GRACE_pred edge cases."""
        torch.manual_seed(42)
        model = GRACE_pred(
            num_layer=1,  # Single layer
            in_channels=2,
            emb_dim=64,
            num_additional_feat=0,
            num_graph_tasks=1,
            node_embedding_output="last",
            drop_ratio=0.0,  # No dropout
            graph_pooling="max",
            gnn_type="gcn",
            activation=torch.nn.functional.relu
        )
        
        data = create_simple_graph()
        output = model(data)
        assert output[0].shape == torch.Size([1, 1])
        assert not torch.isnan(output[0]).any()

    def test_grace_pred_batch_processing(self):
        """Test GRACE_pred with batched graphs."""
        torch.manual_seed(42)
        model = GRACE_pred(
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
        
        # Create multiple graphs
        graph1 = create_simple_graph()
        graph2 = create_simple_graph()
        
        # Process individually
        output1 = model(graph1)
        output2 = model(graph2)
        
        assert output1[0].shape == torch.Size([1, 1])
        assert output2[0].shape == torch.Size([1, 1])
        assert not torch.isnan(output1[0]).any()
        assert not torch.isnan(output2[0]).any()


class TestGRACEIntegration:
    """Integration tests for GRACE components."""

    def test_encoder_to_grace_model(self):
        """Test integration from encoder to GRACE model."""
        torch.manual_seed(42)
        
        # Create encoder
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GCNConv,
            k=2
        )
        
        # Create GRACE model
        grace_model = GRACEModel(
            encoder=encoder,
            num_hidden=64,
            num_proj_hidden=64,
            tau=0.5
        )
        
        # Test forward pass
        data = create_simple_graph()
        output = grace_model(data.x, data.edge_index, data.edge_weight)
        assert output.shape == (3, 64)
        assert not torch.isnan(output).any()

    def test_grace_model_to_grace_pred(self):
        """Test integration from GRACE model to GRACE_pred."""
        torch.manual_seed(42)
        
        # Create GRACE model
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=GCNConv,
            k=2
        )
        
        grace_model = GRACEModel(
            encoder=encoder,
            num_hidden=64,
            num_proj_hidden=64,
            tau=0.5
        )
        
        # Create GRACE_pred with same encoder architecture
        grace_pred = GRACE_pred(
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
        
        # Test both models
        data = create_simple_graph()
        
        grace_output = grace_model(data.x, data.edge_index, data.edge_weight)
        pred_output = grace_pred(data)
        
        assert grace_output.shape == (3, 64)
        assert pred_output[0].shape == torch.Size([1, 1])
        assert not torch.isnan(grace_output).any()
        assert not torch.isnan(pred_output[0]).any()

    def test_different_gnn_types_integration(self):
        """Test integration with different GNN types."""
        torch.manual_seed(42)
        
        for gnn_type in ["gcn", "gin", "gat"]:
            # Create encoder
            if gnn_type == "gin":
                base_model = GINConv
            elif gnn_type == "gat":
                base_model = GATConv
            else:
                base_model = GCNConv
            
            encoder = Encoder(
                in_channels=2,
                out_channels=64,
                activation=torch.nn.functional.relu,
                base_model=base_model,
                k=2
            )
            
            # Create GRACE model
            grace_model = GRACEModel(
                encoder=encoder,
                num_hidden=64,
                num_proj_hidden=64,
                tau=0.5
            )
            
            # Create GRACE_pred
            grace_pred = GRACE_pred(
                num_layer=2,
                in_channels=2,
                emb_dim=64,
                num_additional_feat=0,
                num_graph_tasks=1,
                node_embedding_output="last",
                drop_ratio=0.1,
                graph_pooling="max",
                gnn_type=gnn_type,
                activation=torch.nn.functional.relu
            )
            
            # Test both models
            data = create_simple_graph()
            if gnn_type == "gin":
                # use existing edge_attr
                grace_output = grace_model(data.x, data.edge_index, edge_attr=data.edge_attr)
            else:
                grace_output = grace_model(data.x, data.edge_index, edge_attr=data.edge_attr)
            
            pred_output = grace_pred(data)
            
            assert grace_output.shape == (3, 64)
            assert pred_output[0].shape == torch.Size([1, 1])
            assert not torch.isnan(grace_output).any()
            assert not torch.isnan(pred_output[0]).any()


if __name__ == "__main__":
    pytest.main([__file__]) 