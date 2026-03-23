"""
DEEP-GOMS v2 — Loss Functions
==============================
All 7 loss terms, each importable independently.

    L_total = L_ICB
            + α1·L_graph
            + α2·L_batch
            + α3·L_causal
            + α4·L_microbe
            + α5·L_exercise   [NEW — Fix #8: SupCon with protocol labels]
            + α6·L_mimic      [Fix #7: now wired into training loop]

Optimiser: AdamW (lr=1e-4, weight_decay=1e-5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# L_ICB — Binary cross-entropy for ICB response prediction
# ──────────────────────────────────────────────────────────────────────────────

class ICBLoss(nn.Module):
    """
    Standard BCE with optional class-imbalance weighting.
    pos_weight: ratio of negatives to positives (e.g. 3.0 if 75% non-responders).
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pw", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets.float(),
                                                   pos_weight=self.pw)


# ──────────────────────────────────────────────────────────────────────────────
# L_graph — Graph structure preservation (Laplacian regularisation)
# ──────────────────────────────────────────────────────────────────────────────

class GraphStructureLoss(nn.Module):
    """
    Encourages co-expressed (positive adjacency) nodes to have similar
    embeddings and antagonistic (negative) nodes to have dissimilar ones.

        L_graph = Σ_{(u,v): A_pos>0} ||h_u - h_v||²
                - λ · Σ_{(u,v): A_neg>0} ||h_u - h_v||²
    """

    def __init__(self, margin: float = 1.0, neg_lambda: float = 0.5):
        super().__init__()
        self.margin     = margin
        self.neg_lambda = neg_lambda

    def forward(self, H: torch.Tensor,
                A_pos: torch.Tensor, A_neg: torch.Tensor) -> torch.Tensor:
        # H: (B, N, d_gnn),  A_pos/A_neg: (N, N)
        diffs = H.unsqueeze(2) - H.unsqueeze(1)           # (B, N, N, d)
        dist2 = (diffs ** 2).sum(-1)                       # (B, N, N)

        pos_loss = (A_pos.unsqueeze(0) * dist2).sum() / (A_pos.sum() + 1e-8)
        neg_loss = (A_neg.unsqueeze(0) * dist2).sum() / (A_neg.sum() + 1e-8)

        return pos_loss - self.neg_lambda * neg_loss


# ──────────────────────────────────────────────────────────────────────────────
# L_batch — Batch-effect regularisation (MMD-based)
# ──────────────────────────────────────────────────────────────────────────────

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor,
                sigma: float = 1.0) -> torch.Tensor:
    diff = x.unsqueeze(1) - y.unsqueeze(0)                 # (N, M, d)
    return torch.exp(-(diff ** 2).sum(-1) / (2 * sigma**2))


class BatchEffectLoss(nn.Module):
    """
    Maximum Mean Discrepancy between embeddings from two different cohorts.
    Minimising this encourages cohort-invariant representations.
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, Z_a: torch.Tensor, Z_b: torch.Tensor) -> torch.Tensor:
        K_aa = _rbf_kernel(Z_a, Z_a, self.sigma).mean()
        K_bb = _rbf_kernel(Z_b, Z_b, self.sigma).mean()
        K_ab = _rbf_kernel(Z_a, Z_b, self.sigma).mean()
        return K_aa + K_bb - 2 * K_ab


# ──────────────────────────────────────────────────────────────────────────────
# L_causal — Causal directionality (gut → intratumoral microbiome flow)
# ──────────────────────────────────────────────────────────────────────────────

class CausalDirectionalityLoss(nn.Module):
    """
    Penalises violations of the assumed causal order:
        gut microbiome (Z_M)  →  intratumoral microbiome (Z_Y)  →  immune response

    Implemented as a cosine-similarity alignment loss that encourages Z_M to
    be predictive of Z_Y but not vice versa (asymmetric).

    A learned causal projection W_causal: R^128 → R^128 is applied to Z_M
    before computing the target alignment with Z_Y.
    """

    def __init__(self, d: int = 128):
        super().__init__()
        self.W_causal = nn.Linear(d, d, bias=False)

    def forward(self, Z_M: torch.Tensor, Z_Y: torch.Tensor) -> torch.Tensor:
        Z_M_proj = self.W_causal(Z_M)
        # Maximise alignment (minimise negative cosine similarity)
        return -F.cosine_similarity(Z_M_proj, Z_Y.detach(), dim=-1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# L_microbe — Biological relevance regularisation
# ──────────────────────────────────────────────────────────────────────────────

class MicrobeBiologicalRegLoss(nn.Module):
    """
    Encourages the model to weight biologically relevant taxa (SCFA producers,
    known ICB-associated species) more heavily than background taxa.

    Prior weights prior_w (n_taxa,) are pre-specified from literature / ONCOBIOME
    and used to scale a L2 penalty on the encoder's input layer weights.
    """

    def __init__(self, encoder_input_layer: nn.Linear,
                 prior_w: Optional[torch.Tensor] = None):
        super().__init__()
        self.layer = encoder_input_layer
        if prior_w is not None:
            self.register_buffer("prior_w", prior_w)
        else:
            self.prior_w = None

    def forward(self) -> torch.Tensor:
        W = self.layer.weight                              # (out, n_taxa)
        feature_norms = W.norm(dim=0)                     # (n_taxa,)
        if self.prior_w is not None:
            # Penalise deviation from biologically-informed importance prior
            return F.mse_loss(feature_norms,
                              self.prior_w.to(feature_norms.device))
        return feature_norms.mean()


# ──────────────────────────────────────────────────────────────────────────────
# L_exercise — Exercise-immune contrastive alignment  [NEW — Fix #8]
# ──────────────────────────────────────────────────────────────────────────────

class ExerciseContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss (Khosla et al. 2020) on Z_E embeddings.

    Positive pairs  : same exercise protocol class (e.g. HIIT patients)
    Negative pairs  : different protocol or sedentary controls

    exercise_protocol_labels : (B,) integer class labels
        0 = sedentary control
        1 = MICT (moderate-intensity continuous training)
        2 = HIIT (high-intensity interval training)
        3 = resistance training
        (extend as needed)

    temperature: 0.07 — standard for biological contrastive tasks.

    Fix #8: contrastive pair construction is now explicit.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, Z_E: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Z_E    : (B, 64)
        labels : (B,)   integer exercise protocol class
        """
        B = Z_E.size(0)
        Z_norm = F.normalize(Z_E, dim=-1)             # unit sphere

        # Pairwise cosine similarity matrix
        sim = torch.mm(Z_norm, Z_norm.T) / self.temperature   # (B, B)

        # Mask: positive = same label, excluding self
        label_mat = labels.unsqueeze(1) == labels.unsqueeze(0)  # (B, B) bool
        pos_mask  = label_mat & ~torch.eye(B, dtype=torch.bool, device=Z_E.device)
        neg_mask  = ~label_mat

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=Z_E.device)

        # Log-sum-exp over all negatives per anchor
        # Numerically stable: subtract row max before exp
        sim_max, _ = sim.max(dim=1, keepdim=True)
        exp_sim    = torch.exp(sim - sim_max.detach())

        log_denom = torch.log(
            (exp_sim * neg_mask.float()).sum(dim=1, keepdim=True) + 1e-8
        )
        log_pos   = sim - sim_max.detach() - log_denom          # (B, B)

        # Average over positive pairs per anchor
        loss = -(log_pos * pos_mask.float()).sum(dim=1) / (pos_mask.float().sum(dim=1) + 1e-8)
        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# L_mimic — Microbial-tumour antigen mimicry  [Fix #7: now computed explicitly]
# ──────────────────────────────────────────────────────────────────────────────

class AntigenMimicryLoss(nn.Module):
    """
    Penalises divergence between microbial peptide-space embeddings (from Z_M)
    and tumour neoantigen embeddings (from Z_P) for known mimicry pairs.

    In practice this requires a pre-built mimicry pair index (from BLAST or
    sequence similarity searches against IEDB). Pairs are passed as index
    tensors at call time.

    If no mimicry pairs are available for a batch, the loss returns 0.

    Fix #7: this loss was defined in the loss table but never computed in the
    previous algorithm pseudocode.  It is now wired into DEEPGOMSTrainer.
    """

    def __init__(self, d: int = 128, projection_dim: int = 64):
        super().__init__()
        # Project both Z_M and Z_P to the same comparison space
        self.proj_M = nn.Linear(d, projection_dim)
        self.proj_P = nn.Linear(64, projection_dim)   # Z_P is 64-dim

    def forward(self, Z_M: torch.Tensor, Z_P: torch.Tensor,
                mimicry_idx_m: Optional[torch.Tensor] = None,
                mimicry_idx_p: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Z_M           : (B, 128)
        Z_P           : (B,  64)
        mimicry_idx_m : (K,)  indices into the batch for microbial side of pairs
        mimicry_idx_p : (K,)  indices into the batch for neoantigen side
        """
        if mimicry_idx_m is None or len(mimicry_idx_m) == 0:
            return torch.tensor(0.0, device=Z_M.device)

        Z_m_sel = self.proj_M(Z_M[mimicry_idx_m])     # (K, proj_dim)
        Z_p_sel = self.proj_P(Z_P[mimicry_idx_p])     # (K, proj_dim)

        # Maximise similarity for known mimicry pairs
        return 1.0 - F.cosine_similarity(Z_m_sel, Z_p_sel, dim=-1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Combined Loss
# ──────────────────────────────────────────────────────────────────────────────

class DEEPGOMSTotalLoss(nn.Module):
    """
    L_total = L_ICB + α1·L_graph + α2·L_batch + α3·L_causal
            + α4·L_microbe + α5·L_exercise + α6·L_mimic

    Alpha defaults are conservative starting points; tune with Optuna or grid
    search on a validation cohort.
    """

    def __init__(
        self,
        alpha1: float = 0.1,    # graph
        alpha2: float = 0.05,   # batch
        alpha3: float = 0.1,    # causal
        alpha4: float = 0.05,   # microbe
        alpha5: float = 0.1,    # exercise  [NEW]
        alpha6: float = 0.05,   # mimic
        pos_weight: float = 1.0,
        microbe_encoder_layer: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.alpha = dict(graph=alpha1, batch=alpha2, causal=alpha3,
                         microbe=alpha4, exercise=alpha5, mimic=alpha6)

        self.L_ICB      = ICBLoss(pos_weight)
        self.L_graph    = GraphStructureLoss()
        self.L_batch    = BatchEffectLoss()
        self.L_causal   = CausalDirectionalityLoss()
        self.L_microbe  = MicrobeBiologicalRegLoss(microbe_encoder_layer) \
                          if microbe_encoder_layer else None
        self.L_exercise = ExerciseContrastiveLoss()
        self.L_mimic    = AntigenMimicryLoss()

    def forward(self, batch: dict) -> dict:
        """
        Expects batch to contain:
            logits, targets,
            H_nodes, A_pos, A_neg,       (for L_graph)
            Z_a, Z_b,                    (for L_batch — two cohort subsets)
            Z_M, Z_Y,                    (for L_causal)
            Z_E, exercise_labels,        (for L_exercise)
            Z_P, mimicry_idx_m/p         (for L_mimic)
        """
        losses = {}

        losses["L_ICB"] = self.L_ICB(batch["logits"], batch["targets"])

        losses["L_graph"] = self.L_graph(
            batch["H_nodes"], batch["A_pos"], batch["A_neg"]
        ) if "H_nodes" in batch else torch.tensor(0.0)

        losses["L_batch"] = self.L_batch(
            batch["Z_a"], batch["Z_b"]
        ) if "Z_a" in batch and batch["Z_a"].size(0) > 1 else torch.tensor(0.0)

        losses["L_causal"] = self.L_causal(batch["Z_M"], batch["Z_Y"])

        losses["L_microbe"] = self.L_microbe() \
            if self.L_microbe is not None else torch.tensor(0.0)

        losses["L_exercise"] = self.L_exercise(
            batch["Z_E"], batch["exercise_labels"]
        ) if "exercise_labels" in batch else torch.tensor(0.0)

        losses["L_mimic"] = self.L_mimic(
            batch["Z_M"], batch["Z_P"],
            batch.get("mimicry_idx_m"), batch.get("mimicry_idx_p")
        )

        L_total = (
            losses["L_ICB"]
            + self.alpha["graph"]    * losses["L_graph"]
            + self.alpha["batch"]    * losses["L_batch"]
            + self.alpha["causal"]   * losses["L_causal"]
            + self.alpha["microbe"]  * losses["L_microbe"]
            + self.alpha["exercise"] * losses["L_exercise"]
            + self.alpha["mimic"]    * losses["L_mimic"]
        )
        losses["L_total"] = L_total
        return losses
