"""
DEEP-GOMS v2 — Evaluation Metrics
====================================
All 8 evaluation metrics from Table 18, including the new exercise subgroup AUC.

    AUROC              sklearn.metrics.roc_auc_score
    AUPRC              sklearn.metrics.average_precision_score
    Brier Score        sklearn.metrics.brier_score_loss
    F1 / MCC           f1_score, matthews_corrcoef
    LODO-CV AUC        LeaveOneCohortOut cross-validation
    Exercise Sub-AUC   Stratified by VO2max quartile / protocol class  [NEW]
    SHAP Consistency   Spearman correlation of SHAP ranks across cohorts
    Calibration Curve  sklearn.calibration.calibration_curve
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr


# ──────────────────────────────────────────────────────────────────────────────
# Standard metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_standard_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUROC":       float(roc_auc_score(y_true, y_prob)),
        "AUPRC":       float(average_precision_score(y_true, y_prob)),
        "Brier":       float(brier_score_loss(y_true, y_prob)),
        "F1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC":         float(matthews_corrcoef(y_true, y_pred)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LODO cross-validation AUC
# ──────────────────────────────────────────────────────────────────────────────

def lodo_cv_auc(
    predict_fn,            # callable: (Z_train, y_train, Z_test) → y_prob
    Z_all: np.ndarray,
    y_all: np.ndarray,
    cohort_ids: np.ndarray,
) -> dict:
    """
    Leave-One-Cohort-Out cross-validation.
    predict_fn should fit on train split and return probabilities for test.

    Returns per-cohort AUC and mean ± std.
    """
    cohorts = np.unique(cohort_ids)
    aucs    = {}
    for c in cohorts:
        test_mask  = cohort_ids == c
        train_mask = ~test_mask
        if y_all[test_mask].sum() == 0 or y_all[test_mask].sum() == test_mask.sum():
            continue  # skip cohorts with only one class
        y_prob_test = predict_fn(Z_all[train_mask], y_all[train_mask],
                                 Z_all[test_mask])
        aucs[c] = float(roc_auc_score(y_all[test_mask], y_prob_test))

    auc_vals = list(aucs.values())
    return {
        "LODO_per_cohort":    aucs,
        "LODO_mean_AUC":      float(np.mean(auc_vals)),
        "LODO_std_AUC":       float(np.std(auc_vals)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Exercise Subgroup AUC  [NEW]
# ──────────────────────────────────────────────────────────────────────────────

def exercise_subgroup_auc(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    vo2max:    Optional[np.ndarray] = None,
    protocol:  Optional[np.ndarray] = None,
) -> dict:
    """
    Stratified AUC by:
        (a) VO2max quartile   (if vo2max provided)
        (b) Exercise protocol class (if protocol provided)
            0=sedentary, 1=MICT, 2=HIIT, 3=resistance

    Assesses exercise-specific predictive lift relative to full-cohort AUC.
    """
    results = {}
    full_auc = float(roc_auc_score(y_true, y_prob))
    results["full_AUROC"] = full_auc

    # VO2max quartile stratification
    if vo2max is not None:
        quartiles = np.quantile(vo2max, [0.25, 0.5, 0.75])
        q_labels  = np.digitize(vo2max, quartiles)   # 0–3
        for q in range(4):
            mask = q_labels == q
            if mask.sum() < 10 or y_true[mask].sum() == 0:
                continue
            try:
                auc = float(roc_auc_score(y_true[mask], y_prob[mask]))
                results[f"AUC_VO2max_Q{q+1}"] = auc
            except ValueError:
                pass

    # Protocol stratification
    protocol_names = {0: "sedentary", 1: "MICT", 2: "HIIT", 3: "resistance"}
    if protocol is not None:
        for cls, name in protocol_names.items():
            mask = protocol == cls
            if mask.sum() < 10 or y_true[mask].sum() == 0:
                continue
            try:
                auc = float(roc_auc_score(y_true[mask], y_prob[mask]))
                results[f"AUC_{name}"] = auc
                results[f"lift_{name}"] = auc - full_auc
            except ValueError:
                pass

    return results


# ──────────────────────────────────────────────────────────────────────────────
# SHAP Consistency (cross-cohort Spearman correlation of feature ranks)
# ──────────────────────────────────────────────────────────────────────────────

def shap_consistency(
    shap_per_cohort: Dict[str, np.ndarray],
    top_k: int = 50,
) -> dict:
    """
    For each pair of cohorts, compute Spearman correlation between the
    top-k mean|SHAP| feature rankings.

    shap_per_cohort : dict {cohort_name: (N_i, n_features) shap array}
    Returns mean Spearman ρ across all cohort pairs.
    """
    cohort_names  = list(shap_per_cohort.keys())
    mean_abs_shap = {c: np.abs(v).mean(axis=0)
                     for c, v in shap_per_cohort.items()}

    rhos = []
    pairs = []
    for i in range(len(cohort_names)):
        for j in range(i + 1, len(cohort_names)):
            ci, cj = cohort_names[i], cohort_names[j]
            rho, p = spearmanr(mean_abs_shap[ci], mean_abs_shap[cj])
            rhos.append(rho)
            pairs.append((ci, cj))

    return {
        "SHAP_consistency_mean_rho": float(np.mean(rhos)) if rhos else np.nan,
        "SHAP_consistency_per_pair": dict(zip(
            [f"{a}_vs_{b}" for a, b in pairs], rhos
        )),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Calibration curve
# ──────────────────────────────────────────────────────────────────────────────

def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Returns fraction_of_positives, mean_predicted_value for a reliability plot,
    plus Expected Calibration Error (ECE).
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # ECE: weighted mean absolute deviation between predicted and observed
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(len(bin_edges) - 1):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece     += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)

    return {
        "calibration_frac_pos":  frac_pos.tolist(),
        "calibration_mean_pred": mean_pred.tolist(),
        "ECE":                   float(ece),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation report
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    cohort_ids: Optional[np.ndarray] = None,
    vo2max:    Optional[np.ndarray] = None,
    protocol:  Optional[np.ndarray] = None,
    shap_per_cohort: Optional[dict] = None,
    predict_fn = None,
    Z_all:     Optional[np.ndarray] = None,
) -> dict:
    """
    Master evaluation function — mirrors Table 18.
    Only computes optional metrics when the required inputs are provided.
    """
    results = compute_standard_metrics(y_true, y_prob)

    if cohort_ids is not None and predict_fn is not None and Z_all is not None:
        results.update(lodo_cv_auc(predict_fn, Z_all, y_true, cohort_ids))

    if vo2max is not None or protocol is not None:
        results.update(exercise_subgroup_auc(y_true, y_prob, vo2max, protocol))

    if shap_per_cohort is not None:
        results.update(shap_consistency(shap_per_cohort))

    results.update(compute_calibration(y_true, y_prob))

    return results
