"""
DEEP-GOMS v2 — SHAP Explainability
====================================
Aggregated SHAP replaces gradient saliency as the GOMS feature selection
criterion.

    GOMS_features = top_k( mean|SHAP_xgb| + mean|SHAP_deep| + mean|SHAP_svm|,
                            k=50 )

Fix #6 : KernelSHAP on Z_prime would be O(d²) per sample at d=256+.
         nsamples=200 and l1_reg='auto' are enforced here.
         Option to run KernelSHAP on the top-100 XGB-SHAP features only.
"""

import numpy as np
import torch
import shap
from typing import List, Optional, Tuple
import warnings


# ──────────────────────────────────────────────────────────────────────────────
# TreeSHAP for XGBoost  (exact, fast)
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_xgb(
    xgb_model,
    Z_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, shap.Explainer]:
    """
    Returns:
        shap_values : (N, n_features)  signed SHAP values
        explainer   : shap.TreeExplainer for expected_value access
    """
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(Z_test)

    if isinstance(shap_values, list):
        # Multi-class: take class-1 (responder) SHAP values
        shap_values = shap_values[1]

    return shap_values, explainer


# ──────────────────────────────────────────────────────────────────────────────
# DeepSHAP for PyTorch ensemble members
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_deep(
    model: torch.nn.Module,
    Z_test: torch.Tensor,
    background: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    DeepSHAP via shap.DeepExplainer.

    model      : a DeepEnsembleMember (or any nn.Module with scalar output)
    Z_test     : (N_test, z_dim)  torch.Tensor
    background : (N_bg,  z_dim)  reference distribution (e.g. training mean)

    Returns shap_values : (N_test, z_dim)
    """
    model.eval()
    model.to(device)
    bg   = background.to(device)
    test = Z_test.to(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer   = shap.DeepExplainer(model, bg)
        shap_values = explainer.shap_values(test)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return np.array(shap_values)


# ──────────────────────────────────────────────────────────────────────────────
# KernelSHAP for SVM  (Fix #6)
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_svm(
    svm_predict_fn,
    Z_test: np.ndarray,
    Z_background: np.ndarray,
    n_bg_samples: int = 100,
    n_shap_samples: int = 200,
    top_xgb_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    KernelSHAP for the SVM pipeline.

    Fix #6: KernelSHAP is O(n_features²) per sample.
        - n_bg_samples=100     : subsample background (fast summary)
        - n_shap_samples=200   : approximate SHAP computation
        - l1_reg='auto'        : sparsity-inducing regularisation
        - top_xgb_features     : if provided, run KernelSHAP only on the
                                  top-100 features from XGBoost SHAP (much faster)

    svm_predict_fn : callable, e.g. svm_pipeline.predict_proba
    Returns shap_values : (N_test, n_features)  zeros-padded if subset used
    """
    background = shap.sample(Z_background, n_bg_samples)

    if top_xgb_features is not None:
        # Reduce dimensionality before KernelSHAP  (Fix #6 optimisation)
        idx = top_xgb_features[:100]

        def reduced_predict(X_sub):
            X_full = np.zeros((X_sub.shape[0], Z_test.shape[1]))
            X_full[:, idx] = X_sub
            return svm_predict_fn(X_full)

        bg_sub   = background[:, idx]
        test_sub = Z_test[:, idx]

        explainer   = shap.KernelExplainer(reduced_predict, bg_sub)
        sv_sub      = explainer.shap_values(test_sub,
                                             nsamples=n_shap_samples,
                                             l1_reg="auto",
                                             silent=True)
        if isinstance(sv_sub, list):
            sv_sub = sv_sub[1]

        # Pad back to full feature space
        shap_values = np.zeros_like(Z_test, dtype=float)
        shap_values[:, idx] = sv_sub
    else:
        explainer   = shap.KernelExplainer(svm_predict_fn, background)
        shap_values = explainer.shap_values(Z_test,
                                             nsamples=n_shap_samples,
                                             l1_reg="auto",
                                             silent=True)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    return np.array(shap_values)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregated SHAP → GOMS signature selection
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_shap(
    shap_xgb:  np.ndarray,
    shap_deep: np.ndarray,
    shap_svm:  np.ndarray,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    shap_agg = w1·mean|SHAP_xgb| + w2·mean|SHAP_deep| + w3·mean|SHAP_svm|

    Inputs  : each (N, n_features)
    Returns : (n_features,)  aggregated importance scores
    """
    w1, w2, w3 = weights
    agg = (
        w1 * np.abs(shap_xgb).mean(axis=0)
        + w2 * np.abs(shap_deep).mean(axis=0)
        + w3 * np.abs(shap_svm).mean(axis=0)
    )
    return agg


def select_goms_features(
    shap_agg: np.ndarray,
    feature_names: List[str],
    k: int = 50,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    GOMS = top_k( shap_agg, k=50 )

    Returns:
        top_indices  : (k,)        feature indices in Z_prime
        top_names    : list[str]   feature names
        top_scores   : (k,)        importance values
    """
    top_idx    = np.argsort(shap_agg)[::-1][:k]
    top_names  = [feature_names[i] for i in top_idx]
    top_scores = shap_agg[top_idx]
    return top_idx, top_names, top_scores


# ──────────────────────────────────────────────────────────────────────────────
# Patient-level SHAP report
# ──────────────────────────────────────────────────────────────────────────────

def patient_shap_report(
    shap_xgb:      np.ndarray,
    expected_value: float,
    patient_idx:   int,
    Z_test:        np.ndarray,
    feature_names: List[str],
    top_k: int = 20,
) -> dict:
    """
    Returns a structured dict for a single patient's SHAP waterfall data.
    Suitable for downstream plotting or report generation.
    """
    sv     = shap_xgb[patient_idx]                         # (n_features,)
    order  = np.argsort(np.abs(sv))[::-1][:top_k]

    return {
        "patient_idx":    patient_idx,
        "base_value":     expected_value,
        "prediction":     float(expected_value + sv.sum()),
        "top_features":   [feature_names[i] for i in order],
        "top_shap_values": sv[order].tolist(),
        "top_feature_values": Z_test[patient_idx][order].tolist(),
    }
