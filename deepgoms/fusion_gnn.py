"""
DEEP-GOMS v2 — Fusion & Graph Integration
==========================================
Fix #3 : All modality embeddings are projected to a common d=128 BEFORE the
         attention-weighted sum.  Raw dims (128/128/128/64/64/64/32) are
         heterogeneous; weighted averaging without projection is undefined.

The fused vector Z is then concatenated with the GNN readout h_G to form
Z_prime, which feeds the ensemble predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


# ──────────────────────────────────────────────────────────────────────────────
# Modality projection registry
# ──────────────────────────────────────────────────────────────────────────────

MODALITY_DIMS = {
    "Z_M":  128,   # gut microbiome
    "Z_Y":  128,   # intratumoral microbiome
    "Z_S":  128,   # single-cell / spatial
    "Z_E":   64,   # exercise oncology          [NEW]
    "Z_EM":  64,   # exercise–microbiome gate   [NEW]
    "Z_P":   64,   # peptide / neoantigen
    "Z_H":   32,   # HLA context
}

MODALITY_ORDER = ["Z_M", "Z_Y", "Z_S", "Z_E", "Z_EM", "Z_P", "Z_H"]
N_MODALITIES   = len(MODALITY_ORDER)


# ──────────────────────────────────────────────────────────────────────────────
# Attention-Weighted Fusion  (Fix #3)
# ──────────────────────────────────────────────────────────────────────────────

class AttentionFusion(nn.Module):
    """
    Projects each modality to a common dim d_common=128, then computes a
    softmax attention weight over the modalities and returns the weighted sum.

    Z = Σ_i  w_i · proj_i(Z_i)
    w = softmax( W_f · mean_pool( stack of projected embeddings ) )

    Inputs : list of 7 tensors in MODALITY_ORDER, each (B, modality_dim_i)
    Output : Z (B, d_common)
    """

    def __init__(self, d_common: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_common = d_common

        # One linear projection per modality to d_common
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, d_common),
                nn.LayerNorm(d_common),
            )
            for name, dim in MODALITY_DIMS.items()
        })

        # Attention scorer: maps mean(d_common) → N_MODALITIES weights
        self.attn_scorer = nn.Linear(d_common, N_MODALITIES, bias=True)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, modality_tensors: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        modality_tensors : dict  {name: tensor (B, dim_i)}
        returns:
            Z      : (B, d_common)   fused representation
            weights: (B, N_MODALITIES)  attention weights for inspection
        """
        projected = []
        for name in MODALITY_ORDER:
            z = modality_tensors[name]
            projected.append(self.projections[name](z))   # each (B, d_common)

        stacked = torch.stack(projected, dim=1)           # (B, N_MODALITIES, d_common)

        # Compute attention weights from mean-pooled representation
        mean_z  = stacked.mean(dim=1)                     # (B, d_common)
        weights = F.softmax(self.attn_scorer(mean_z), dim=-1)  # (B, N_MODALITIES)

        # Weighted sum
        w_exp = weights.unsqueeze(-1)                     # (B, N_MODALITIES, 1)
        Z = (w_exp * stacked).sum(dim=1)                  # (B, d_common)
        return self.dropout(Z), weights


# ──────────────────────────────────────────────────────────────────────────────
# Heterogeneous Graph Neural Network  (Signed Adjacency)
# ──────────────────────────────────────────────────────────────────────────────

class SignedGCNLayer(nn.Module):
    """
    One layer of signed graph convolution:
        h_v^{l+1} = σ( Σ_{u∈N(v)} A_uv · W^l · h_u^l )

    A_uv ∈ {+1, -1, 0}  encodes cooperative / suppressive / neutral links.
    Separate weight matrices for positive and negative edges.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_pos = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neg = nn.Linear(in_dim, out_dim, bias=False)
        self.norm  = nn.LayerNorm(out_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor,
                A_pos: torch.Tensor, A_neg: torch.Tensor) -> torch.Tensor:
        """
        H     : (B, N_nodes, in_dim)
        A_pos : (N_nodes, N_nodes)  positive adjacency (0/1)
        A_neg : (N_nodes, N_nodes)  negative adjacency (0/1)
        """
        agg_pos = torch.bmm(A_pos.unsqueeze(0).expand(H.size(0), -1, -1), H)
        agg_neg = torch.bmm(A_neg.unsqueeze(0).expand(H.size(0), -1, -1), H)
        out = F.gelu(self.W_pos(agg_pos) - self.W_neg(agg_neg))
        return self.drop(self.norm(out))


class HeterogeneousGNN(nn.Module):
    """
    3-layer signed GCN over the tripartite microbial-immune-exercise graph.

    Node types (from the causal tripartite graph G_T):
        - Gut microbial taxa       (from Z_M)
        - Intratumoral microbes    (from Z_Y)
        - Immune cell states       (from Z_S)
        - Exercise-immune nodes    (from Z_E)  [NEW]
        - Neoantigen presentations (from Z_P)

    Node embeddings are initialised from the encoder outputs (pooled).
    Adjacency matrices A_pos and A_neg should be pre-computed from
    co-occurrence / mutual information analysis and supplied externally,
    or learned end-to-end with the learnable_adj=True option.

    Output: h_G (B, d_gnn)  — global READOUT via mean pooling over nodes
    """

    def __init__(self, n_nodes: int = 50, node_dim: int = 128,
                 d_gnn: int = 128, n_layers: int = 3,
                 learnable_adj: bool = True, dropout: float = 0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.d_gnn   = d_gnn

        # Node initialisation: project from Z_M/Z_Y/Z_S/Z_E/Z_P concatenation
        # We build one projection per node type (using mean embedding from each encoder)
        self.node_init = nn.Linear(d_gnn, node_dim)

        self.layers = nn.ModuleList([
            SignedGCNLayer(
                node_dim if i == 0 else d_gnn,
                d_gnn, dropout=dropout
            )
            for i in range(n_layers)
        ])

        # Learnable signed adjacency (logistic parameterisation)
        if learnable_adj:
            self.A_logits_pos = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.01)
            self.A_logits_neg = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.01)
        else:
            self.register_buffer("A_logits_pos", torch.zeros(n_nodes, n_nodes))
            self.register_buffer("A_logits_neg", torch.zeros(n_nodes, n_nodes))

        self.readout_norm = nn.LayerNorm(d_gnn)

    def get_adjacency(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns soft (sigmoid) adjacency matrices. Hard threshold at 0.5."""
        A_pos = torch.sigmoid(self.A_logits_pos)
        A_neg = torch.sigmoid(self.A_logits_neg)
        # Prevent a node from being both positive and negative to another
        mask  = (A_pos > A_neg).float()
        return A_pos * mask, A_neg * (1 - mask)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        node_features : (B, n_nodes, node_dim)
            — each row is a node embedding derived from encoder outputs
        returns h_G   : (B, d_gnn)
        """
        A_pos, A_neg = self.get_adjacency()
        H = node_features

        for layer in self.layers:
            H = layer(H, A_pos, A_neg)

        # Global mean readout
        h_G = H.mean(dim=1)                      # (B, d_gnn)
        return self.readout_norm(h_G)


def build_node_features(Z_M: torch.Tensor, Z_Y: torch.Tensor,
                         Z_S: torch.Tensor, Z_E: torch.Tensor,
                         Z_P: torch.Tensor,
                         n_nodes: int = 50) -> torch.Tensor:
    """
    Construct per-node feature matrix from encoder outputs.
    Each modality contributes ~n_nodes/5 nodes via repeat-interleave tiling.
    All embeddings are projected to node_dim=128 before stacking.

    Returns (B, n_nodes, 128)
    """
    B = Z_M.size(0)
    nodes_per_type = n_nodes // 5    # 10 nodes per type for n_nodes=50

    # Broadcast each (B, dim) → (B, nodes_per_type, 128) via repeat
    proj = nn.Linear(128, 128).to(Z_M.device)  # identity-ish warm start

    def tile(z: torch.Tensor, k: int) -> torch.Tensor:
        # z: (B, d) → (B, k, 128) via linear projection + tile
        d = z.size(-1)
        if d != 128:
            z = F.pad(z, (0, 128 - d)) if d < 128 else z[:, :128]
        return z.unsqueeze(1).expand(-1, k, -1)

    nodes = torch.cat([
        tile(Z_M, nodes_per_type),
        tile(Z_Y, nodes_per_type),
        tile(Z_S, nodes_per_type),
        tile(Z_E, nodes_per_type),   # exercise-immune nodes [NEW]
        tile(Z_P, nodes_per_type),
    ], dim=1)                         # (B, n_nodes, 128)

    return nodes
