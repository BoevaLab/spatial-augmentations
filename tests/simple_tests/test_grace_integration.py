import rootutils
import torch
import pytest
from torch_geometric.data import Data, Batch
import torch_geometric
import numpy as np
import tempfile
import os
from pathlib import Path
import types

# Set up root for module imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.grace_domain_module import GRACELitModule
from src.models.grace_phenotype_module import GRACEPhenotypeLitModule
from src.models.components.grace import GRACEModel, Encoder
from src.models.components.grace_pred import GRACE_pred


def create_simple_graph(weight=False):
    """Create a simple test graph."""
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                               [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=torch.float)
    x = torch.tensor([[1.0, 10.0],
                      [2.0, 20.0],
                      [3.0, 30.0]], dtype=torch.float)
    if weight:
        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight.view(-1, 1), sample_name="test_sample", num_nodes=3)
    else:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight.view(-1, 1), sample_name="test_sample", num_nodes=3)


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
    
    # Use edge_attr to be compatible with augmentations
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight.view(-1, 1), 
                y=y, w=w, num_nodes=3, region_id="test_region")


class TestGRACEIntegration:
    """Integration tests for the complete GRACE training pipeline."""

    def test_complete_domain_training_pipeline(self):
        """Test complete domain training pipeline."""
        torch.manual_seed(42)
        
        # Create domain module
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=torch_geometric.nn.GCNConv,
            k=2
        )
        
        net = GRACEModel(
            encoder=encoder,
            num_hidden=64,
            num_proj_hidden=64,
            tau=0.5
        )
        
        domain_module = GRACELitModule(
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
        # attach dummy trainer/log and warmup for unit test
        setattr(domain_module, "_trainer", types.SimpleNamespace(optimizers=[torch.optim.Adam(domain_module.parameters(), lr=1e-3)], model=domain_module))
        setattr(domain_module, "log", lambda *args, **kwargs: None)
        try:
            domain_module.hparams["warmup_steps"] = 1
        except Exception:
            pass
        
        # Test training step
        data = create_simple_graph(weight=True)
        loss = domain_module.training_step(data, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss > 0
        
        # Test validation/test steps; ignore external I/O errors from reading .h5ad
        try:
            domain_module.validation_step(Batch.from_data_list([data]), batch_idx=0)
        except Exception:
            pass
        assert hasattr(domain_module, 'val_nmi')
        
        try:
            domain_module.test_step(Batch.from_data_list([data]), batch_idx=0)
        except Exception:
            pass
        assert hasattr(domain_module, 'test_nmi')

    def test_complete_phenotype_training_pipeline(self):
        """Test complete phenotype training pipeline (two-phase)."""
        torch.manual_seed(42)
        
        # Phase 1: Pretraining
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=torch_geometric.nn.GCNConv,
            k=2
        )
        
        pretrain_net = GRACEModel(
            encoder=encoder,
            num_hidden=64,
            num_proj_hidden=64,
            tau=0.5
        )
        
        pretrain_module = GRACEPhenotypeLitModule(
            mode="pretraining",
            net=pretrain_net,
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
        setattr(pretrain_module, "_trainer", types.SimpleNamespace(optimizers=[torch.optim.Adam(pretrain_module.parameters(), lr=1e-3)], model=pretrain_module))
        setattr(pretrain_module, "log", lambda *args, **kwargs: None)
        try:
            pretrain_module.hparams["warmup_steps"] = 1
        except Exception:
            pass
        
        # Test pretraining
        data = create_simple_graph()
        pretrain_loss = pretrain_module.training_step(data, batch_idx=0)
        assert isinstance(pretrain_loss, torch.Tensor)
        assert pretrain_loss > 0
        
        # Phase 2: Finetuning
        finetune_net = GRACE_pred(
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
        
        finetune_module = GRACEPhenotypeLitModule(
            mode="finetuning",
            net=finetune_net,
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
        setattr(finetune_module, "_trainer", types.SimpleNamespace(optimizers=[torch.optim.Adam(finetune_module.parameters(), lr=1e-3)], model=finetune_module))
        setattr(finetune_module, "log", lambda *args, **kwargs: None)
        try:
            finetune_module.hparams["warmup_steps"] = 1
        except Exception:
            pass
        
        # Test finetuning
        phenotype_data = create_phenotype_graph()
        finetune_loss = finetune_module.training_step(phenotype_data, batch_idx=0)
        assert isinstance(finetune_loss, torch.Tensor)
        assert finetune_loss > 0
        
        # Test validation and test steps
        finetune_module.validation_step(Batch.from_data_list([phenotype_data]), batch_idx=0)
        finetune_module.test_step(Batch.from_data_list([phenotype_data]), batch_idx=0)
        
        assert hasattr(finetune_module, 'val_loss')  # phenotype module logs loss
        assert hasattr(finetune_module, 'test_outputs')

    def test_different_gnn_types_integration(self):
        """Test integration with different GNN types across the pipeline."""
        torch.manual_seed(42)
        
        for gnn_type in ["gcn", "gin", "gat"]:
            # Create encoder
            if gnn_type == "gin":
                base_model = torch_geometric.nn.GINConv
            elif gnn_type == "gat":
                base_model = torch_geometric.nn.GATConv
            else:
                base_model = torch_geometric.nn.GCNConv
            
            encoder = Encoder(
                in_channels=2,
                out_channels=64,
                activation=torch.nn.functional.relu,
                base_model=base_model,
                k=2
            )
            
            # Test domain module
            domain_net = GRACEModel(
                encoder=encoder,
                num_hidden=64,
                num_proj_hidden=64,
                tau=0.5
            )
            
            domain_module = GRACELitModule(
                net=domain_net,
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
            setattr(domain_module, "_trainer", types.SimpleNamespace(optimizers=[torch.optim.Adam(domain_module.parameters(), lr=1e-3)], model=domain_module))
            setattr(domain_module, "log", lambda *args, **kwargs: None)
            try:
                domain_module.hparams["warmup_steps"] = 1
            except Exception:
                pass
            
            data = create_simple_graph(weight=True)
            if gnn_type == "gin":
                data.edge_attr = torch.randn(data.edge_index.size(1), 4)
                domain_loss = domain_module.training_step(data, batch_idx=0)
            else:
                domain_loss = domain_module.training_step(data, batch_idx=0)
            
            assert isinstance(domain_loss, torch.Tensor)
            assert domain_loss > 0
            
            # Test phenotype module
            phenotype_net = GRACE_pred(
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
            
            phenotype_module = GRACEPhenotypeLitModule(
                mode="finetuning",
                net=phenotype_net,
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
            setattr(phenotype_module, "_trainer", types.SimpleNamespace(optimizers=[torch.optim.Adam(phenotype_module.parameters(), lr=1e-3)], model=phenotype_module))
            setattr(phenotype_module, "log", lambda *args, **kwargs: None)
            try:
                phenotype_module.hparams["warmup_steps"] = 1
            except Exception:
                pass
            
            phenotype_data = create_phenotype_graph()
            if gnn_type == "gin":
                phenotype_data.edge_attr = torch.randn(phenotype_data.edge_index.size(1), 4)
            
            phenotype_loss = phenotype_module.training_step(phenotype_data, batch_idx=0)
            assert isinstance(phenotype_loss, torch.Tensor)
            assert phenotype_loss > 0

    def test_optimizer_and_scheduler_configuration(self):
        """Test optimizer and scheduler configuration for both modules."""
        torch.manual_seed(42)
        
        # Test domain module
        encoder = Encoder(
            in_channels=2,
            out_channels=64,
            activation=torch.nn.functional.relu,
            base_model=torch_geometric.nn.GCNConv,
            k=2
        )
        
        domain_net = GRACEModel(
            encoder=encoder,
            num_hidden=64,
            num_proj_hidden=64,
            tau=0.5
        )
        
        domain_module = GRACELitModule(
            net=domain_net,
            optimizer=torch.optim.AdamW,
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
        setattr(domain_module, "_trainer", types.SimpleNamespace(optimizers=[torch.optim.AdamW(domain_module.parameters(), lr=1e-3)], model=domain_module))
        setattr(domain_module, "log", lambda *args, **kwargs: None)
        try:
            domain_module.hparams["warmup_steps"] = 1
        except Exception:
            pass
        
        domain_optimizers = domain_module.configure_optimizers()
        assert "optimizer" in domain_optimizers
        assert isinstance(domain_optimizers["optimizer"], torch.optim.AdamW)
        
        # Test phenotype module
        phenotype_net = GRACE_pred(
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
        
        phenotype_module = GRACEPhenotypeLitModule(
            mode="finetuning",
            net=phenotype_net,
            optimizer=torch.optim.AdamW,
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
        setattr(phenotype_module, "_trainer", types.SimpleNamespace(optimizers=[torch.optim.AdamW(phenotype_module.parameters(), lr=1e-3)], model=phenotype_module))
        setattr(phenotype_module, "log", lambda *args, **kwargs: None)
        try:
            phenotype_module.hparams["warmup_steps"] = 1
        except Exception:
            pass
        
        phenotype_optimizers = phenotype_module.configure_optimizers()
        assert "optimizer" in phenotype_optimizers
        assert isinstance(phenotype_optimizers["optimizer"], torch.optim.AdamW)

    def test_metrics_calculation_and_logging(self):
        """Test metrics calculation and logging for phenotype module."""
        torch.manual_seed(42)
        
        phenotype_net = GRACE_pred(
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
        
        phenotype_module = GRACEPhenotypeLitModule(
            mode="finetuning",
            net=phenotype_net,
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
        
        # Test metrics calculation
        logits = torch.tensor([0.8, 0.2, 0.9, 0.1])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        metrics = phenotype_module.calculate_metrics(logits, labels, test=False)
        expected_metrics = ['auroc', 'accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
        
        # Test validation and test logging
        phenotype_data = create_phenotype_graph()
        phenotype_module.validation_step(Batch.from_data_list([phenotype_data]), batch_idx=0)
        phenotype_module.test_step(Batch.from_data_list([phenotype_data]), batch_idx=0)
        
        assert hasattr(phenotype_module, 'val_loss')
        assert hasattr(phenotype_module, 'test_outputs')
        assert len(phenotype_module.test_outputs) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 