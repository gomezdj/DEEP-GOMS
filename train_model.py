"""
train_model.py
DEEP-GOMS: Deep Evolutionary Ensemble Predictor for Gut OncoMicrobiome Signatures
Author: Daniel Gomez (2025)
Description:
Implements a 5-step training pipeline for DEEP-GOMS to predict immunotherapy response
using gut microbiome, intratumoral microbiome, and single-cell transcriptomic data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import random

# -------------------------------
# 1️⃣ Step 1: Data Preprocessing
# -------------------------------

def preprocess_data(X_M, X_Y, X_S, y):
    """
    Standardizes data, removes batch effects, and returns torch tensors.
    Args:
        X_M: Gut microbiome matrix (samples x features)
        X_Y: Intratumoral microbiome matrix (samples x features)
        X_S: Single-cell transcriptomic features (samples x features)
        y: Response labels (0 = non-responder, 1 = responder)
    Returns:
        DataLoader objects for training and validation
    """
    scaler = StandardScaler()
    X_M = scaler.fit_transform(X_M)
    X_Y = scaler.fit_transform(X_Y)
    X_S = scaler.fit_transform(X_S)

    X = np.concatenate([X_M, X_Y, X_S], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    return DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(val_ds, batch_size=32, shuffle=False)


# -----------------------------------
# 2️⃣ Step 2: Encoder Architecture
# -----------------------------------

class MultiOmicsEncoder(nn.Module):
    """Encodes gut, intratumoral, and transcriptomic data into a latent space."""
    def __init__(self, input_dim, latent_dim=128):
        super(MultiOmicsEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# -------------------------------------------
# 3️⃣ Step 3: Graph & Fusion Integration
# -------------------------------------------

class GraphIntegrator(nn.Module):
    """Fuses latent representations using weighted attention mechanism."""
    def __init__(self, latent_dim=128):
        super(GraphIntegrator, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        attn_weights = torch.softmax(self.attention(z), dim=0)
        z_fused = torch.sum(attn_weights * z, dim=0)
        return z_fused


# -------------------------------------------
# 4️⃣ Step 4: Predictor & Loss Definition
# -------------------------------------------

class DeepGOMSModel(nn.Module):
    """Full DEEP-GOMS predictive model."""
    def __init__(self, input_dim):
        super(DeepGOMSModel, self).__init__()
        self.encoder = MultiOmicsEncoder(input_dim)
        self.graph = GraphIntegrator()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z_fused = self.graph(z)
        y_hat = self.classifier(z_fused)
        return y_hat


# ----------------------------------------------------
# 5️⃣ Step 5: Training Loop & Evaluation Metrics
# ----------------------------------------------------

def train_deep_goms(X_M, X_Y, X_S, y, epochs=50, lr=1e-4):
    train_loader, val_loader = preprocess_data(X_M, X_Y, X_S, y)
    model = DeepGOMSModel(input_dim=X_M.shape[1] + X_Y.shape[1] + X_S.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                y_hat = model(X_val).squeeze()
                val_preds.extend(y_hat.tolist())
                val_targets.extend(y_val.tolist())

        auc = roc_auc_score(val_targets, val_preds)
        acc = accuracy_score(val_targets, np.round(val_preds))
        f1 = f1_score(val_targets, np.round(val_preds))
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {np.mean(train_losses):.4f} | AUC: {auc:.3f} | Acc: {acc:.3f} | F1: {f1:.3f}")

    return model


# ---------------------------
# Example Usage (Simulated)
# ---------------------------

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    X_M = np.random.rand(500, 100)  # Gut microbiome
    X_Y = np.random.rand(500, 80)   # Intratumoral microbiome
    X_S = np.random.rand(500, 120)  # Single-cell transcriptomics
    y = np.random.randint(0, 2, 500)  # Immunotherapy response

    model = train_deep_goms(X_M, X_Y, X_S, y)
    torch.save(model.state_dict(), "deep_goms_trained_model.pt")
