"""BagMLP: seed-bag of standardized Torch MLPs over C2's 180 features.

Replaces the C2 tree ensemble (ExtraTrees+HistGBM). Trees collapse to near-flat
predict_proba on the validator-sanitized live feed (OOD) -> random within-batch
ranking -> median live score. A bag of MLPs with inputs standardized on the
train mean/std EXTRAPOLATES onto the OOD live batches, preserving a discriminative
ordering. Each member early-stops on validation LOSS (not AP) so the net learns a
spread-preserving surface instead of an AP-frozen flat one. Serve mean probability;
within-batch rank happens in detector.py.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from poker44_model.mlp_member import TorchMLPClassifier


class BagMLP:
    def __init__(self, seed=0, n_members=5, n_epochs=120, patience=15,
                 hidden_sizes=(512, 256, 128), dropout=0.3):
        self.seed = int(seed)
        self.n_members = int(n_members)
        self.n_epochs = int(n_epochs)
        self.patience = int(patience)
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = float(dropout)
        self.members = []

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).reshape(-1)
        self.members = []
        for m in range(self.n_members):
            s = self.seed * 100 + m + 1
            Xtr, Xval, ytr, yval = train_test_split(
                X, y, test_size=0.20, random_state=s, stratify=y)
            clf = TorchMLPClassifier(
                hidden_sizes=self.hidden_sizes, dropout=self.dropout,
                lr=1e-3, weight_decay=1e-4, n_epochs=self.n_epochs,
                batch_size=256, patience=self.patience, class_weight=True,
                seed=s, verbose=False)
            clf.fit(Xtr, ytr, X_val=Xval, y_val=yval)
            self.members.append(clf)
        return self

    def predict_proba(self, X):
        ps = np.mean([c.predict_proba(X)[:, 1] for c in self.members], axis=0)
        return np.stack([1.0 - ps, ps], axis=1)
