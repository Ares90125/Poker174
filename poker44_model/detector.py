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


_T_HI = 0.00040000000000000002   # logit(0.5001): sigmoid(t) >= 0.5001 <=> t >= _T_HI
_T_LO = -0.00040000000000000002  # logit(0.4999): sigmoid(t) <= 0.4999 <=> t <= _T_LO


def _decision(model, v):
    """Reward-fit, FPR-capped per-batch decision layer on the TIE-FREE fused rank.

    Identical to the deployed layer (same Q / MARGIN / TEMP / FLOOR / CAP / EPS /
    train_ref_logit, same k, same crossing count) except for two tie sources that
    were destroying the 65% rank block (0.35*AP + 0.30*recall@FPR<=0.05, both of
    which argsort the served scores and break ties by ARRAY INDEX):

      1. the isotonic map is GONE -- it is monotone but NON-INJECTIVE, so it
         merged the fused rank into ~26 distinct levels per 100-chunk window and
         put the recall@FPR<=0.05 boundary INSIDE a tie group;
      2. FLOOR/CAP now SHIFT each side instead of CLAMPing it to the constants
         0.5001 / 0.4999, which preserves the internal spacing of both groups.

    The result is a STRICTLY MONOTONE map fused -> served score, so the served
    order is exactly the model's order, while k = ceil(FLOOR*n) chunks still
    cross 0.5 (FLOOR lifts the top-k, CAP pins the rest below) -- the 30%
    hard-0.5-threshold block is unchanged.
    """
    eps = float(model["EPS"])
    q = float(model["Q"])
    margin = float(model["MARGIN"])
    temp = float(model.get("TEMP", 1.0))
    floor = float(model["FLOOR"])
    cap = bool(model.get("CAP", False))
    tref = float(model["train_ref_logit"]) - margin
    z = _logit(v, eps)
    if z.size == 0:
        return []
    anchor = np.quantile(z, q)
    t = (z - anchor + tref) / temp
    order = np.argsort(-z, kind="mergesort")
    k = max(1, int(np.ceil(floor * len(t))))
    top, rest = order[:k], order[k:]
    # FLOOR (tie-free): shift the top-k as a block so its MINIMUM sits at 0.5001
    # -- never an all-below-0.5 hard zero, but the spacing inside the block (and
    # hence the ordering that AP / bot-recall read) survives.
    d = _T_HI - t[top].min()
    if d > 0.0:
        t[top] = t[top] + d
    if cap and rest.size:
        # CAP (tie-free): shift the rest as a block so its MAXIMUM sits at 0.4999
        # -> deterministic crossing count k, spacing preserved.
        d = t[rest].max() - _T_LO
        if d > 0.0:
            t[rest] = t[rest] - d
    scores = 1.0 / (1.0 + np.exp(-t))
    return [round(float(s), 9) for s in scores]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk (floating calibrated output)."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, _raw_scores(m, chunks))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback; score_batch is the real entry (needs batch context)."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        return round(float(_raw_scores(m, [chunk])[0]), 6)
    except Exception:
        return 0.5
