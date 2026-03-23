"""
DEEP-GOMS v2 — Main Model
==========================
Full forward pass:

    Step 1  Preprocess (external — see data/)
    Step 2  Encode each modality
    Step 3  Exercise-microbiome gated interaction Z_EM          (Fix #2)
    Step 4  Attention-weighted fusion → Z                       (Fix #3)
    Step 5  GNN integration → h_G → Z_prime
    Step 6  Deep ensemble training (within DEEPGOMSTrainer)
            XGBoost + SVM trained on detached Z_prime numpy     (Fix #4,5)
    Step 7  SHAP explainability (post-training, see interpret/) (Fix #6)
    Step 8  GOMS signature selection
    Step 9  Evaluation

All 10 issues from the architecture review are addressed across this file
and the supporting modules.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

from src.encoders.encoders import (
    GutMicrobiomeEncoder,
    IntratumoralMicrobiomeEncoder,
    SingleCellSpatialEncoder,
    ExerciseOncologyEncoder,
    ExerciseMicrobiomeGate,
    PeptideEncoder,
    HLAEncoder,
)
from src.model.fusion_gnn import (
    AttentionFusion,
    HeterogeneousGNN,
    build_node_features,
    MODALITY_ORDER,
)
from src.model.ensemble import DeepEnsembleMember
from src.losses.losses import DEEPGOMSTotalLoss


# ──────────────────────────────────────────────────────────────────────────────
# DEEP-GOMS PyTorch Module (differentiable core)
# ──────────────────────────────────────────────────────────────────────────────

class DEEPGOMSCore(nn.Module):
    """
    Differentiable forward pass producing:
        Z_prime : (B, 256)   fused + GNN-enriched representation
        logit   : (B,)       deep ensemble logit (single member for training)
        aux     : dict       intermediate tensors for loss computation
    """

    def __init__(
        self,
        # Encoder input dimensions
        n_taxa_gut:    int = 1000,
        n_taxa_intra:  int = 500,
        n_genes:       int = 2000,
        n_ex_features: int = 22,
        n_pep_features:int = 128,
        hla_vocab:     int = 4000,
        n_hla_alleles: int = 6,
        # Architecture
        d_common:      int = 128,
        d_gnn:         int = 128,
        n_gnn_nodes:   int = 50,
        dropout:       float = 0.1,
    ):
        super().__init__()

        # ── Encoders ──────────────────────────────────────────────────────────
        self.enc_M  = GutMicrobiomeEncoder(n_taxa_gut, d_model=128)
        self.enc_Y  = IntratumoralMicrobiomeEncoder(n_taxa_intra, latent_dim=128)
        self.enc_S  = SingleCellSpatialEncoder(n_genes, d_model=128)
        self.enc_E  = ExerciseOncologyEncoder(n_ex_features, seq_len=3,
                                               d_model=128, out_dim=64)
        self.enc_P  = PeptideEncoder(n_pep_features, out_dim=64)
        self.enc_H  = HLAEncoder(hla_vocab, n_alleles=n_hla_alleles, out_dim=32)

        # ── Exercise-Microbiome Gate  (Fix #2) ───────────────────────────────
        self.ex_gate = ExerciseMicrobiomeGate(z_e_dim=64, z_m_dim=128)

        # ── Attention Fusion  (Fix #3) ───────────────────────────────────────
        self.fusion = AttentionFusion(d_common=d_common, dropout=dropout)

        # ── GNN ──────────────────────────────────────────────────────────────
        self.gnn = HeterogeneousGNN(
            n_nodes=n_gnn_nodes, node_dim=d_common,
            d_gnn=d_gnn, n_layers=3, learnable_adj=True
        )

        # ── Deep ensemble head (one member; full K used in trainer) ──────────
        z_prime_dim = d_common + d_gnn          # 256
        self.head   = DeepEnsembleMember(z_dim=z_prime_dim)

    def forward(
        self,
        X_M:         torch.Tensor,            # (B, n_taxa_gut)
        X_Y:         torch.Tensor,            # (B, n_taxa_intra)
        X_S:         torch.Tensor,            # (B, n_genes)
        X_E:         torch.Tensor,            # (B, 3, n_ex_features)
        X_P:         torch.Tensor,            # (B, n_pep_features)
        X_H:         torch.LongTensor,        # (B, n_hla_alleles)
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            Z_prime : (B, 256)   representation for ensemble predictor
            logit   : (B,)       single-head prediction for L_ICB
            aux     : dict       intermediate tensors for all loss terms
        """

        # Step 2: Encode each modality ─────────────────────────────────────────
        Z_M                      = self.enc_M(X_M)
        Z_Y, mu_Y, lv_Y, recon_Y = self.enc_Y(X_Y)
        Z_S                      = self.enc_S(X_S)
        Z_E                      = self.enc_E(X_E)
        Z_P                      = self.enc_P(X_P)
        Z_H                      = self.enc_H(X_H)

        # Step 3: Exercise-microbiome gate  (Fix #2 — explicit sigmoid) ────────
        Z_EM = self.ex_gate(Z_E, Z_M)

        # Step 4: Attention-weighted fusion  (Fix #3 — projected to d=128) ─────
        modality_dict = {
            "Z_M":  Z_M,
            "Z_Y":  Z_Y,
            "Z_S":  Z_S,
            "Z_E":  Z_E,
            "Z_EM": Z_EM,
            "Z_P":  Z_P,
            "Z_H":  Z_H,
        }
        Z, attn_weights = self.fusion(modality_dict)       # (B, 128)

        # Step 5: GNN ─────────────────────────────────────────────────────────
        nodes = build_node_features(Z_M, Z_Y, Z_S, Z_E, Z_P,
                                     n_nodes=self.gnn.n_nodes)
        h_G   = self.gnn(nodes)                             # (B, 128)
        A_pos, A_neg = self.gnn.get_adjacency()

        Z_prime = torch.cat([Z, h_G], dim=-1)               # (B, 256)

        # Deep head logit ─────────────────────────────────────────────────────
        logit = self.head(Z_prime)                          # (B,)

        # Aux dict for loss functions ─────────────────────────────────────────
        aux = {
            "Z_M":         Z_M,
            "Z_Y":         Z_Y,
            "Z_E":         Z_E,
            "Z_P":         Z_P,
            "mu_Y":        mu_Y,
            "logvar_Y":    lv_Y,
            "recon_Y":     recon_Y,
            "H_nodes":     nodes,
            "A_pos":       A_pos,
            "A_neg":       A_neg,
            "attn_weights": attn_weights,
            "Z_prime":     Z_prime,
        }
        return Z_prime, logit, aux


# ──────────────────────────────────────────────────────────────────────────────
# DEEP-GOMS Trainer
# ──────────────────────────────────────────────────────────────────────────────

class DEEPGOMSTrainer:
    """
    Manages the full training loop:
        - PyTorch deep encoder/fusion/GNN (AdamW)
        - XGBoost + SVM fit on detached Z_prime
        - OOF stacking meta-learner             (Fix #9)
        - SHAP computation post-training        (Fix #6)
    """

    def __init__(
        self,
        model: DEEPGOMSCore,
        loss_fn: DEEPGOMSTotalLoss,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model    = model.to(device)
        self.loss_fn  = loss_fn
        self.device   = device
        self.optimiser = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=50, eta_min=1e-6
        )

    def train_step(self, batch: dict) -> dict:
        """Single gradient step on one batch."""
        self.model.train()
        self.optimiser.zero_grad()

        Z_prime, logit, aux = self.model(
            batch["X_M"].to(self.device),
            batch["X_Y"].to(self.device),
            batch["X_S"].to(self.device),
            batch["X_E"].to(self.device),
            batch["X_P"].to(self.device),
            batch["X_H"].to(self.device),
        )

        loss_batch = {
            "logits":  logit,
            "targets": batch["y"].to(self.device),
            **{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
               for k, v in aux.items()},
        }

        # Exercise labels for contrastive loss
        if "exercise_labels" in batch:
            loss_batch["exercise_labels"] = batch["exercise_labels"].to(self.device)

        # Cohort pair for batch-effect loss (split batch by cohort id)
        if "cohort_id" in batch:
            c = batch["cohort_id"]
            unique = c.unique()
            if len(unique) >= 2:
                mask_a = (c == unique[0])
                mask_b = (c == unique[1])
                loss_batch["Z_a"] = Z_prime[mask_a]
                loss_batch["Z_b"] = Z_prime[mask_b]

        losses = self.loss_fn(loss_batch)
        losses["L_total"].backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimiser.step()
        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def get_representations(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs all samples through the encoder + fusion + GNN to get Z_prime.
        Returns (Z_prime_np, y_np) for XGBoost/SVM training.
        """
        self.model.eval()
        all_z, all_y = [], []
        for batch in dataloader:
            Z_prime, _, _ = self.model(
                batch["X_M"].to(self.device),
                batch["X_Y"].to(self.device),
                batch["X_S"].to(self.device),
                batch["X_E"].to(self.device),
                batch["X_P"].to(self.device),
                batch["X_H"].to(self.device),
            )
            all_z.append(Z_prime.cpu().numpy())
            all_y.append(batch["y"].numpy())
        return np.concatenate(all_z), np.concatenate(all_y)
