"""
Attentive Pocket GNN for druggability prediction.

Visualizing the Architecture: Generated using text-to-ascii diagram tool
------------
The model predicts a scalar druggability logit for each pocket graph.

                          ┌─────────────────────────────────┐
                          │  Graph stream (node features x) │
                          └────────────────┬────────────────┘
                                           │
                               NodeEncoder (Lin --> ReLU --> Drop)
                                           │
                              ┌────────────▼─────────────┐
                              │  GAT-1 (4 heads, 128d)   │
                              │  + GraphNorm             │
                              └────────────┬─────────────┘
                              ┌────────────▼─────────────┐
                              │  GAT-2 (4 heads, 128d)   │
                              │  + GraphNorm + residual  │
                              └────────────┬─────────────┘
                              ┌────────────▼─────────────┐
                              │  GAT-3 (1 head,  128d)   │
                              │  + GraphNorm             │
                              └────────────┬─────────────┘
                                     ┌─────┴────┐
                                 mean pool     max pool     sum pool
                                     └─────┬────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │  Global stream (pocket desc. u) │
                          │ GlobalEncoder(Lin -> LN -> ReLU)│
                          └────────────────┬────────────────┘
                                           │
                              concat [mean, max, sum, u_enc]
                                           │
                               FusionMLP (256->128->64->1)
                                           │
                                       logit(s)

Loss: Focal Loss (alpha=0.85, gamma=2.0) for minority-class (druggable) focus.

References
Based on the following papers and techniques:
----------
Veličković et al. (2018) Graph Attention Networks. ICLR.
  https://arxiv.org/abs/1710.10903
Cai et al. (2021) GraphNorm. NeurIPS.
  https://arxiv.org/abs/2009.03294
Lin et al. (2017) Focal Loss for Dense Object Detection. ICCV.
  https://arxiv.org/abs/1708.02002
"""


import torch     # type: ignore[import-untyped, import-not-found]
import torch.nn as nn   # type: ignore[import-untyped, import-not-found]
import torch.nn.functional as F     # type: ignore[import-untyped, import-not-found]
from torch.nn import Dropout, LayerNorm, Linear, Sequential     # type: ignore[import-untyped, import-not-found]
from torch_geometric.nn import (        # type: ignore[import-untyped, import-not-found]
    GATConv,
    GraphNorm,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


# Loss
class FocalLoss(nn.Module):
    """
    Binary focal loss operating on logits.

    Parameters
    ----------
    alpha: float
        Weight for the positive class.  Set > 0.5 when positives are rare.
    gamma: float
        Focusing exponent.  gamma=0 recovers standard BCE; gamma=2 is canonical.
    """

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits).detach()
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return loss.mean()


# Model class (simplified)
class AttentivePocketGNN(nn.Module):
    """
    Multi-head Graph Attention Network for pocket druggability prediction.

    Parameters
    ----------
    node_dim: int
        Dimensionality of node feature vectors (30 in default featurisation).
    edge_dim: int
        Dimensionality of edge feature vectors (4 in default featurisation).
    global_dim: int
        Dimensionality of the global pocket descriptor (11 by default).
    hidden_dim: int
        Hidden layer width.  Must be divisible by `heads`.
    heads: int
        Number of attention heads in GAT-1 and GAT-2.
    dropout: float
        Dropout probability applied after every activation.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        head_dim = hidden_dim // heads

        # ---- Node encoder ---- #
        self.node_encoder = Sequential(
            Linear(node_dim, hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
        )

        # ---- GAT stack ---- #
        self.gat1 = GATConv(
            hidden_dim, head_dim, heads=heads, dropout=dropout, edge_dim=edge_dim
        )
        self.gn1 = GraphNorm(hidden_dim)

        self.gat2 = GATConv(
            hidden_dim, head_dim, heads=heads, dropout=dropout, edge_dim=edge_dim
        )
        self.gn2 = GraphNorm(hidden_dim)

        self.gat3 = GATConv(
            hidden_dim, hidden_dim, heads=1, dropout=dropout, edge_dim=edge_dim
        )
        self.gn3 = GraphNorm(hidden_dim)

        # ---- Global encoder ---- #
        self.global_encoder = Sequential(
            Linear(global_dim, 64),
            LayerNorm(64),
            nn.ReLU(),
            Dropout(dropout),
            Linear(64, 32),
        )

        # ---- Fusion head: mean+max+sum pooled (3×hidden) + global (32) -> 1 ---- #
        fusion_in = hidden_dim * 3 + 32
        self.fusion = Sequential(
            Linear(fusion_in, 256),
            LayerNorm(256),
            nn.ReLU(),
            Dropout(dropout),
            Linear(256, 128),
            LayerNorm(128),
            nn.ReLU(),
            Dropout(dropout),
            Linear(128, 64),
            LayerNorm(64),
            nn.ReLU(),
            Linear(64, 1),
        )

        self.drop = Dropout(dropout)

    def forward(self, batch) -> torch.Tensor:  # noqa: ANN001
        """Return per-graph logits with shape [B]."""
        x, edge_index = batch.x, batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)

        x = self.node_encoder(x)

        x1 = F.relu(self.gn1(self.gat1(x, edge_index, edge_attr), batch=batch.batch))
        x1 = self.drop(x1)

        x2 = F.relu(self.gn2(self.gat2(x1, edge_index, edge_attr), batch=batch.batch))
        x2 = self.drop(x2) + x1  # residual

        x3 = F.relu(self.gn3(self.gat3(x2, edge_index, edge_attr), batch=batch.batch))
        x3 = self.drop(x3)

        # Graph-level readout
        x_mean = global_mean_pool(x3, batch.batch)
        x_max = global_max_pool(x3, batch.batch)
        x_sum = global_add_pool(x3, batch.batch)

        # Global pocket descriptor
        if hasattr(batch, "u"):
            u = batch.u
            if u.dim() == 1 or u.size(0) != batch.num_graphs:
                u = u.view(batch.num_graphs, -1)
        else:
            u = torch.stack([d.u for d in batch.to_data_list()])
        u = u.to(x_mean.device).float()

        fused = torch.cat([x_mean, x_max, x_sum, self.global_encoder(u)], dim=1)
        return self.fusion(fused).squeeze(1)

    @torch.no_grad()
    def attention_weights(self, batch):  # noqa: ANN001
        """Return edge indices and head-averaged attention from GAT-1."""
        x = self.node_encoder(batch.x)
        _, (ei, attn) = self.gat1(
            x,
            batch.edge_index,
            getattr(batch, "edge_attr", None),
            return_attention_weights=True,
        )
        return ei, attn.mean(dim=-1)
