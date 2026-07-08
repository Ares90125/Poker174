"""Poker44 bot detector — MLP-bag over C2's 180 sanitization-invariant features.

TREE-COLLAPSE FIX. The v5_sani C2 model (ExtraTrees + HistGradientBoosting
soft-vote) collapses to a near-flat predict_proba on the validator-sanitized live
feed: those tree ensembles do not extrapolate off the benchmark support, so live
batches (a shifted, deeper, call-heavier population) get squashed to nearly one
value -> random within-batch ranking -> median live reward.

This model replaces the tree ensemble with a **bag of standardized Torch MLPs**
over the SAME 180 features (mlp_bag.BagMLP / mlp_member.TorchMLPClassifier). Inputs
are standardized on the train mean/std (critical for OOD extrapolation), each
member early-stops on validation LOSS (not AP) so it learns a spread-preserving
surface, and 5 seed members are averaged. Offline double-gate vs C2:
  Gate A (benchmark GroupKFold reward, true labels): 0.839 mean vs C2 0.836 (3 seeds)
  Gate B (live dup-proxy Spearman): +0.35 mean vs C2 +0.043 (11-12/12 batches positive)
Gate A is preserved (not the DA mirage, which had a flat Gate A) and Gate B lifts
sharply -> the live ordering is genuinely more discriminative.

IMPORTANT — inference does NOT sanitize. Live chunks arrive already sanitized by
the validator (prepare_hand_for_miner runs validator-side, per hand). Only TRAINING
sanitizes raw benchmark hands. Output = within-batch rank (matches the ranking reward).
"""
from __future__ import annotations

import os

import numpy as np

try:  # bound CPU threads so batched predict stays fast
    import torch
    torch.set_num_threads(int(os.environ.get("POKER44_TORCH_THREADS", "4")))
except Exception:
    pass

import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
    return _MODEL


def _rank_normalize(vals):
    n = len(vals)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = round(pos / (n - 1), 6)
    return out


def _raw_scores(model, chunks):
    # Live chunks are already sanitized by the validator; featurize as-is.
    rows = []
    for c in chunks:
        feats = chunk_features(c)          # compute the feature set ONCE per chunk
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return model.predict_proba(np.array(rows, dtype=float))[:, 1]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk, ranked within the batch."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        return _rank_normalize(list(_raw_scores(_model(), chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk model probability (fallback; batch path is score_batch)."""
    try:
        if not chunk:
            return 0.5
        return round(float(_raw_scores(_model(), [chunk])[0]), 6)
    except Exception:
        return 0.5
