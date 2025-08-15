# This is the GRACE model from the original implementation.
# https://github.com/CRIPAC-DIG/GRACE/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GNN, NUM_EDGE_TYPE


class Encoder(torch.nn.Module):
    """Graph encoder for GRACE model that mirrors the GNN in gnn.py.

    Constructor kept compatible with previous Encoder API, but implementation
    delegates to the self-defined `GNN` from gnn.py.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation,
        base_model,
        num_feat: int = 0,
        num_node_type: int = 30,
        k: int = 2,
        node_embedding_output: str = "last",
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1 (expects first column to be node type id)")
        if k < 1:
            raise ValueError("k (number of layers) must be >= 1")

        self.activation = activation
        self.num_feat = (in_channels - 1) if (num_feat == 0) else num_feat

        gnn_type = self._infer_gnn_type(base_model)

        # Mirror GNN configuration
        self.gnn = GNN(
            num_layer=k,
            num_node_type=num_node_type,
            num_feat=self.num_feat,
            emb_dim=out_channels,
            node_embedding_output=node_embedding_output,
            drop_ratio=0.25,
            gnn_type=gnn_type,
        )

    def _infer_gnn_type(self, base_model) -> str:
        name = getattr(base_model, "__name__", str(base_model)).lower()
        if "gin" in name:
            return "gin"
        if "gat" in name:
            return "gat"
        if "sage" in name:
            return "graphsage"
        if "gcn" in name:
            return "gcn"
        # default to gin if unknown
        return "gin"

    def _sanitize_edge_attr(self, edge_index: torch.Tensor, edge_weight: torch.Tensor = None, edge_attr: torch.Tensor = None) -> torch.Tensor:
        # Prefer provided edge_attr, then edge_weight; if none, create zeros
        attr = edge_attr if edge_attr is not None else edge_weight
        if attr is None:
            num_edges = edge_index.size(1)
            return edge_index.new_zeros((num_edges, 1)).to(dtype=torch.float32)

        if attr.dim() == 1:
            attr = attr.view(-1, 1)
        elif attr.dim() != 2 or attr.size(1) < 1:
            raise ValueError("edge_attr/edge_weight must be 1D [num_edges] or 2D [num_edges, >=1]")

        attr = attr.clone()
        type_col = attr[:, 0]
        if type_col.dtype.is_floating_point:
            type_col = type_col.round()
        type_col = type_col.long().clamp_(0, NUM_EDGE_TYPE - 1)
        attr[:, 0] = type_col.to(attr.dtype)
        return attr

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        # GNN expects first column of x to be node type id and the rest features
        attr = self._sanitize_edge_attr(edge_index, edge_weight, edge_attr)
        return self.gnn(x, edge_index, attr)

    def reset_parameters(self):
        self.gnn.reset_parameters()


class TwoLayerGCNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        from .two_layer_gcn import TwoLayerGCN
        self.gnn = TwoLayerGCN(
            input_size=in_channels,
            hidden_size=out_channels,
            output_size=out_channels,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.gnn(x, edge_index, edge_weight)

    def reset_parameters(self):
        self.gnn.reset_parameters()


class GRACEModel(torch.nn.Module):
    """GRACE (Graph Contrastive Learning) model."""

    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super().__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weight, edge_attr)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            losses.append(
                -torch.log(
                    between_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                    / (
                        refl_sim.sum(1)
                        + between_sim.sum(1)
                        - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                    )
                )
            )

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
