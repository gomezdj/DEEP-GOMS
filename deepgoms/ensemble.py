"""
DEEP-GOMS v2 — Heterogeneous Ensemble Predictor
=================================================
XGBoost + SVM-RBF + K Deep Neural Networks + Stacked Meta-Learner.

Fix #4  : XGBoost uses Python xgboost API throughout (no R mixing).
          early_stopping_rounds=30 added to prevent small-cohort overfitting.
Fix #5  : SVM uses sklearn throughout. CalibratedClassifierCV('prefit') note
          included for large-Z_prime scenarios.
Fix #9  : Meta-learner stacking uses out-of-fold (OOF) predictions on the
          training set via cross_val_predict to prevent leakage.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ──────────────────────────────────────────────────────────────────────────────
# Deep Neural Network (single ensemble member)
# ──────────────────────────────────────────────────────────────────────────────

class DeepEnsembleMember(nn.Module):
    """
    Single MLP member trained on a bootstrapped cohort.
    Input : Z_prime (B, z_dim)   where z_dim = d_common + d_gnn = 128+128 = 256
    Output: logit  (B,)
    """

    def __init__(self, z_dim: int = 256, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64),    nn.LayerNorm(64),    nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)   # (B,)


# ──────────────────────────────────────────────────────────────────────────────
# XGBoost predictor (Fix #4)
# ──────────────────────────────────────────────────────────────────────────────

def build_xgboost(scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    """
    Returns a configured XGBClassifier.

    Fix #4a: Python xgboost API (not R syntax).
    Fix #4b: early_stopping_rounds=30 prevents overfitting on small cohorts.
             Caller must pass eval_set to .fit().
    """
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,   # handles class imbalance
        objective="binary:logistic",
        eval_metric=["auc", "logloss"],
        early_stopping_rounds=30,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SVM predictor (Fix #5)
# ──────────────────────────────────────────────────────────────────────────────

def build_svm(C: float = 1.0, gamma: float = 0.01,
              n_train: int = 5000) -> Pipeline:
    """
    RBF-kernel SVM wrapped in StandardScaler (Z_prime may have varying scale).

    Fix #5a: sklearn throughout — no R e1071.
    Fix #5b: For large Z_prime (n_train > 5000) the internal 5-fold CV in
             SVC(probability=True) is slow. Use CalibratedClassifierCV with
             cv='prefit' after fitting SVC(probability=False) for speed.

    Returns a sklearn Pipeline with scaler → calibrated SVM.
    """
    base_svc = SVC(kernel="rbf", C=C, gamma=gamma, probability=False,
                   random_state=42)

    if n_train > 5_000:
        # Fit SVC without probability, then calibrate post-hoc (faster)
        clf = CalibratedClassifierCV(base_svc, cv="prefit", method="isotonic")
    else:
        # Small cohort: use standard probability=True (Platt scaling via CV)
        clf = SVC(kernel="rbf", C=C, gamma=gamma, probability=True,
                  random_state=42)

    return Pipeline([("scaler", StandardScaler()), ("svm", clf)])


# ──────────────────────────────────────────────────────────────────────────────
# Out-of-fold stacking (Fix #9)
# ──────────────────────────────────────────────────────────────────────────────

def get_oof_predictions(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    xgb_model: xgb.XGBClassifier,
    svm_pipeline: Pipeline,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates out-of-fold (OOF) predictions from XGBoost and SVM for use as
    meta-learner training features.

    Fix #9 : Using in-sample predictions would allow the meta-learner to
             see training-set preds from models trained on that same data →
             leakage and silent overfitting.  cross_val_predict ensures each
             prediction is made by a model that never saw that sample.

    Returns:
        oof_xgb : (N,)  OOF probabilities from XGBoost
        oof_svm : (N,)  OOF probabilities from SVM
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)

    # XGBoost OOF — pass eval_set inside fit_params via cross_val_predict
    # Note: cross_val_predict doesn't support early_stopping natively,
    # so we clone without it for OOF and use full training for final model.
    xgb_oof = xgb.XGBClassifier(
        n_estimators=300,          # fixed rounds for OOF (no val set available)
        max_depth=6, learning_rate=0.01, subsample=0.8,
        objective="binary:logistic", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
    oof_xgb = cross_val_predict(xgb_oof, Z_train, y_train,
                                 cv=cv, method="predict_proba")[:, 1]

    oof_svm = cross_val_predict(svm_pipeline, Z_train, y_train,
                                 cv=cv, method="predict_proba")[:, 1]

    return oof_xgb, oof_svm


# ──────────────────────────────────────────────────────────────────────────────
# Meta-Learner
# ──────────────────────────────────────────────────────────────────────────────

def build_meta_learner() -> LogisticRegression:
    """
    Logistic regression meta-learner over stacked OOF predictions.
    C=1.0 with L2 regularisation; solver='lbfgs' for small feature spaces.
    """
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                               random_state=42)


# ──────────────────────────────────────────────────────────────────────────────
# Full Ensemble Predictor
# ──────────────────────────────────────────────────────────────────────────────

class HeterogeneousEnsemble:
    """
    Orchestrates training and inference for the three-component ensemble.

    Training flow (no leakage):
        1. Fit XGBoost on full Z_train (with early stopping on val set).
        2. Fit SVM on full Z_train.
        3. Generate OOF predictions via 5-fold CV for each model.
        4. Train meta-learner on [oof_xgb, oof_svm, oof_deep] stack.

    Inference flow:
        ŷ = meta_learner( [ŷ_xgb, ŷ_svm, ŷ_deep] )
    """

    def __init__(self, z_dim: int = 256, n_deep: int = 5,
                 scale_pos_weight: float = 1.0):
        self.xgb_model   = build_xgboost(scale_pos_weight)
        self.svm_pipeline = build_svm()
        self.meta_learner = build_meta_learner()
        self.z_dim        = z_dim
        self.n_deep       = n_deep
        self._is_fitted   = False

    def fit(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_val:   np.ndarray,
        y_val:   np.ndarray,
        oof_deep: Optional[np.ndarray] = None,
    ) -> "HeterogeneousEnsemble":
        """
        Z_train, Z_val : (N, z_dim)  numpy arrays (detached from PyTorch graph)
        y_train, y_val : (N,)        integer labels 0/1
        oof_deep       : (N_train,)  OOF preds from deep ensemble (pre-computed)
        """
        # 1. Fit XGBoost with early stopping on validation set
        self.xgb_model.fit(
            Z_train, y_train,
            eval_set=[(Z_val, y_val)],
            verbose=False,
        )

        # 2. Fit SVM  (no early stopping; calibrated post-hoc for large sets)
        n_train = Z_train.shape[0]
        if n_train > 5_000:
            # Two-step: fit base SVC, then calibrate
            base = SVC(kernel="rbf", C=1.0, gamma=0.01,
                       probability=False, random_state=42)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            Z_sc   = scaler.fit_transform(Z_train)
            base.fit(Z_sc, y_train)
            self.svm_pipeline = CalibratedClassifierCV(base, cv="prefit",
                                                        method="isotonic")
            self.svm_pipeline.fit(Z_sc, y_train)
            self._svm_scaler = scaler
        else:
            self.svm_pipeline = build_svm(n_train=n_train)
            self.svm_pipeline.fit(Z_train, y_train)
            self._svm_scaler = None

        # 3. OOF predictions for stacking  (Fix #9)
        oof_xgb, oof_svm = get_oof_predictions(
            Z_train, y_train, self.xgb_model, self.svm_pipeline
        )

        # 4. Stack and fit meta-learner
        if oof_deep is None:
            oof_deep = np.zeros_like(oof_xgb)   # placeholder if deep not ready

        stack_train = np.column_stack([oof_xgb, oof_svm, oof_deep])
        self.meta_learner.fit(stack_train, y_train)
        self._is_fitted = True
        return self

    def predict_proba(
        self,
        Z_test: np.ndarray,
        y_hat_deep: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Returns final stacked probability ŷ_final (N,).
        """
        assert self._is_fitted, "Call .fit() before .predict_proba()"

        y_hat_xgb = self.xgb_model.predict_proba(Z_test)[:, 1]

        if self._svm_scaler is not None:
            Z_sc = self._svm_scaler.transform(Z_test)
            y_hat_svm = self.svm_pipeline.predict_proba(Z_sc)[:, 1]
        else:
            y_hat_svm = self.svm_pipeline.predict_proba(Z_test)[:, 1]

        if y_hat_deep is None:
            y_hat_deep = np.zeros_like(y_hat_xgb)

        stack_test  = np.column_stack([y_hat_xgb, y_hat_svm, y_hat_deep])
        y_hat_final = self.meta_learner.predict_proba(stack_test)[:, 1]
        return y_hat_final

    @property
    def meta_weights(self) -> np.ndarray:
        """Inspect meta-learner coefficients [w_xgb, w_svm, w_deep]."""
        return self.meta_learner.coef_[0]
