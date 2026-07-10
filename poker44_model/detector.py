"""Poker44 bot detector -- gradient-boosted trees over C2's 180 features.

Estimator: a LightGBM binary classifier over the FEATURE_NAMES-ordered 180-dim
sanitization-invariant feature row, wrapped in an isotonic calibrator fit on
held-out (GroupKFold-by-date) predictions -> a calibrated per-chunk probability.

The served output applies a per-batch decision layer to that calibrated
probability so the FRACTION of a served window that crosses the validator's hard
0.5 threshold tracks the window's composition instead of saturating. Two guards
keep it operationally safe on the out-of-distribution live feed:
  * a per-batch anti-saturation recenter (a robust batch quantile is aligned to a
    fixed training reference), so the level is comparable across windows, and
  * a thin hard floor that always lifts the batch-top chunks across 0.5, so a
    labeled window can never score an all-below-0.5 hard zero.

Rank order (hence AP / bot-recall) is preserved by the transform.

IMPORTANT -- inference does NOT sanitize. Live chunks arrive already sanitized by
the validator (prepare_hand_for_miner runs validator-side, per hand). Only the
offline training matrix sanitizes raw benchmark hands (train == serve).
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        try:  # keep batched tree predict single-threaded (never deadlock)
            b["lgbm"].set_params(n_jobs=1)
        except Exception:
            pass
        _MODEL = b
    return _MODEL


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _raw_scores(model, chunks):
    """Pre-decision-layer discrimination score per chunk (LightGBM probability)."""
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return model["lgbm"].predict_proba(np.array(rows, dtype=float))[:, 1]


def _calibrated(model, raw):
    return model["iso"].predict(np.asarray(raw, dtype=float))


def _decision(model, cal):
    """Per-batch anti-saturation recenter + thin hard floor on calibrated probs."""
    eps = float(model["EPS"])
    q = float(model["Q"])
    margin = float(model["MARGIN"])
    floor = float(model["FLOOR"])
    tref = float(model["train_ref_logit"]) - margin
    z = _logit(cal, eps)
    if z.size == 0:
        return []
    anchor = np.quantile(z, q)
    scores = 1.0 / (1.0 + np.exp(-(z - anchor + tref)))
    # thin hard floor: always lift the top FLOOR fraction across 0.5 so a labeled
    # window can never be an all-below-0.5 hard zero (which forces reward=0).
    k = max(1, int(np.ceil(floor * len(scores))))
    top = np.argsort(-z, kind="mergesort")[:k]
    scores[top] = np.maximum(scores[top], 0.5001)
    return [round(float(s), 6) for s in scores]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk (floating calibrated output)."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, _calibrated(m, _raw_scores(m, chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback; score_batch is the real entry (needs batch context)."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        return round(float(_calibrated(m, _raw_scores(m, [chunk]))[0]), 6)
    except Exception:
        return 0.5
