"""
DEEP-GOMS v2 — Modality Encoders
=================================
Six modality encoders feeding into the attention-weighted fusion layer.

Dimensions (fixed contract for the fusion layer):
    Z_M : (B, 128)   gut microbiome      — CNN/Transformer
    Z_Y : (B, 128)   intratumoral micro  — VAE (reparameterised)
    Z_S : (B, 128)   single-cell/spatial — Transformer/MLP
    Z_E : (B,  64)   exercise oncology   — Temporal Transformer [NEW]
    Z_P : (B,  64)   neoantigen/peptide  — MLP
    Z_H : (B,  32)   HLA allelic context — Embedding + MLP

All encoders project to their output dim; the fusion layer projects
everything to a common d=128 before computing attention weights.

Fix #2  — ExerciseEncoder: sigmoid gate is explicit, W_EM dims are (64, 192).
Fix #10 — d_model=128 is divisible by heads=4; assertion guards this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Gut Microbiome Encoder  (Z_M, output 128-dim)
# ──────────────────────────────────────────────────────────────────────────────

class GutMicrobiomeEncoder(nn.Module):
    """
    1-D CNN + Transformer for OTU / metagenomic abundance vectors.
    Input  : (B, n_taxa)  — log-normalised relative abundance
    Output : (B, 128)
    """

    def __init__(self, n_taxa: int = 1000, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.input_proj = nn.Sequential(
            nn.Linear(n_taxa, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_taxa)
        h = self.input_proj(x).unsqueeze(1)          # (B, 1, 128)
        h = self.transformer(h)                       # (B, 1, 128)
        h = h.squeeze(1)                              # (B, 128)
        return self.out_norm(h)


# ──────────────────────────────────────────────────────────────────────────────
# Intratumoral Microbiome Encoder  (Z_Y, output 128-dim)
# ──────────────────────────────────────────────────────────────────────────────

class IntratumoralMicrobiomeEncoder(nn.Module):
    """
    Variational Autoencoder for intratumoral strain-level profiles.
    Returns both the reparameterised sample Z_Y and (mu, logvar) for the
    KL term in the total loss.

    Input  : (B, n_taxa_intra)
    Output : Z_Y (B, 128), mu (B, 128), logvar (B, 128)
    """

    def __init__(self, n_taxa: int = 500, latent_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_taxa, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.LayerNorm(256), nn.GELU(),
        )
        self.mu_head     = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, 512),        nn.GELU(),
            nn.Linear(512, n_taxa),
        )

    def reparameterise(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h      = self.encoder(x)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h)
        z      = self.reparameterise(mu, logvar)
        recon  = self.decoder(z)
        return z, mu, logvar, recon


# ──────────────────────────────────────────────────────────────────────────────
# Single-Cell / Spatial Encoder  (Z_S, output 128-dim)
# ──────────────────────────────────────────────────────────────────────────────

class SingleCellSpatialEncoder(nn.Module):
    """
    Transformer + MLP for scRNAseq or CODEX Phenocycler feature vectors.
    Expects a per-patient aggregated embedding (e.g. pseudobulk mean).

    Input  : (B, n_genes)
    Output : (B, 128)
    """

    def __init__(self, n_genes: int = 2000, d_model: int = 128,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.proj = nn.Sequential(
            nn.Linear(n_genes, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x).unsqueeze(1)
        h = self.transformer(h).squeeze(1)
        return self.mlp(h)


# ──────────────────────────────────────────────────────────────────────────────
# Exercise Oncology & Immunology Encoder  (Z_E, output 64-dim)  [NEW]
# ──────────────────────────────────────────────────────────────────────────────

class ExerciseOncologyEncoder(nn.Module):
    """
    Temporal Transformer for exercise biomarker time-series.

    Input X_E is structured as three time points [pre, during, post] across
    four feature classes:
        - Cardiorespiratory fitness  (VO2max, MET-hrs/wk, 6MWT)
        - Exercise protocol          (HIIT/MICT indicator, duration, freq, intensity)
        - Myokine profile            (IL-6, IL-15, irisin, BDNF, oncostatin-M)
        - Immune response            (NK cytotoxicity, CD8+ TIL, Treg ratio)
        - Metabolic markers          (lactate, cortisol, epinephrine, glucose)
        - Microbiome-exercise shift  (Akkermansia, Bifidobacterium, Lactobacillus Δ)

    n_features : total raw feature count (default 22 across 6 classes)
    seq_len    : time points (default 3 — pre/during/post)

    Fix #2  : W_EM gate for Z_EM is in ExerciseMicrobiomeGate below.
    Fix #10 : d_model=128, heads=4 → 32 per head; assert guards this.

    Output : (B, 64)
    """

    def __init__(self, n_features: int = 22, seq_len: int = 3,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, out_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Project to 64-dim output
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, seq_len, n_features)  — temporal exercise feature matrix
        returns Z_E : (B, 64)
        """
        h = self.input_proj(x) + self.pos_embed   # (B, seq_len, d_model)
        h = self.transformer(h)                    # (B, seq_len, d_model)
        h = h.mean(dim=1)                          # temporal mean pool  (B, d_model)
        return self.out_proj(h)                    # (B, 64)


# ──────────────────────────────────────────────────────────────────────────────
# Exercise-Microbiome Hadamard Gate  (Z_EM)  [NEW — Fix #2]
# ──────────────────────────────────────────────────────────────────────────────

class ExerciseMicrobiomeGate(nn.Module):
    """
    Z_EM = Z_E ⊙ σ( W_EM · [Z_E ; Z_M] )

    Dimensions:
        Z_E  : (B,  64)
        Z_M  : (B, 128)
        concat: (B, 192)
        W_EM : (64, 192)  → gate (B, 64)
        Z_EM : (B,  64)   — same shape as Z_E

    Fix #2 : sigmoid is explicit here (not hidden inside a black-box call).
    """

    def __init__(self, z_e_dim: int = 64, z_m_dim: int = 128):
        super().__init__()
        self.gate = nn.Linear(z_e_dim + z_m_dim, z_e_dim, bias=True)

    def forward(self, z_e: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([z_e, z_m], dim=-1)         # (B, 192)
        gate   = torch.sigmoid(self.gate(concat))       # (B, 64)  — explicit σ
        return z_e * gate                               # (B, 64)  Hadamard product


# ──────────────────────────────────────────────────────────────────────────────
# Peptide / Neoantigen Encoder  (Z_P, output 64-dim)
# ──────────────────────────────────────────────────────────────────────────────

class PeptideEncoder(nn.Module):
    """
    MLP for pre-computed neoantigen binding / physicochemical features.
    Input  : (B, n_pep_features)
    Output : (B, 64)
    """

    def __init__(self, n_features: int = 128, out_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim),    nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# HLA Allelic Context Encoder  (Z_H, output 32-dim)
# ──────────────────────────────────────────────────────────────────────────────

class HLAEncoder(nn.Module):
    """
    Embedding + MLP for patient-specific HLA alleles.
    Alleles are passed as integer indices for a combined HLA vocabulary
    (HLA-A, -B, -C class I; HLA-DR, -DQ class II).

    Input  : (B, n_alleles)   integer allele IDs
    Output : (B, 32)
    """

    def __init__(self, vocab_size: int = 4000, embed_dim: int = 32,
                 n_alleles: int = 6, out_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim * n_alleles, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, allele_ids: torch.Tensor) -> torch.Tensor:
        # allele_ids: (B, n_alleles)
        h = self.embed(allele_ids)          # (B, n_alleles, embed_dim)
        h = h.flatten(1)                    # (B, n_alleles * embed_dim)
        return self.mlp(h)                  # (B, 32)
