"""Poker44 bot detector -- LEARNER_AB: OUR 180-dim C2 surface

Learner (replicated from u89 / hot-poker3-3's PUBLISHED recipe, trained by us on
our own benchmark rows with our own seeds -- their model blob is never used):

    ExtraTrees(n=700, max_depth=9) + RandomForest(n=700, max_depth=9)
    + HistGradientBoosting(max_iter=700, lr=0.03, max_depth=9)
    soft-vote weights 0.45 / 0.25 / 0.30

Why this learner: the unconfounded R2 head-to-head showed live rank block is driven
by ESTIMATOR-VARIANCE REDUCTION, monotone in how much averaging is done --
deep single LGBM (+0.107) -> shallow single (+0.130) -> union ensemble (+0.167)
-> rank-fused C2-180 ensemble (+0.210). This trio is a low-capacity (depth-9
capped), heavily-averaged (1400 trees + 700 boosted iters) vote -- the far end of
that same axis. Bench AP is NOT why it is here (bench AP is a proven live-mirage:
the +0.210 R2 winner has the LOWEST bench AP of our fleet).

Decision layer: reused VERBATIM from the deployed NOISO_XC10 package (isotonic
removed -- it was monotone but NON-INJECTIVE and merged ranks; FLOOR=0.10 so
exactly ceil(0.10*n) chunks cross 0.5; CAP shifts the rest as a block). The map
from vote-rank -> served score is STRICTLY MONOTONE, so the 65% rank block
(AP + recall@FPR<=0.05) is set purely by the model's ordering, and the 30%
threshold block stays pinned (never a hard zero).

IMPORTANT -- inference does NOT sanitize. Live chunks arrive already sanitized by
the validator (prepare_hand_for_miner runs validator-side). Only the offline
training matrix sanitizes raw benchmark hands (train == serve).
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

try:  # MANDATORY for this recipe -- see _predict_all() below
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

_MODEL = None


def _pin_single_thread(est):
    """n_jobs only covers the JOBLIB-parallel members (ET/RF). It does NOT cover
    HistGradientBoosting, which is OpenMP-parallel -- see _predict_all()."""
    for attr in ("n_jobs", "nthread", "thread_count"):
        try:
            est.set_params(**{attr: 1})
        except Exception:
            pass


def _model():
    global _MODEL
    if _MODEL is None:
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        for _n, m, _w in b["members"]:
            _pin_single_thread(m)
        _MODEL = b
    return _MODEL


def _rank01(s):
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def _rows(chunks):
    # chunk_features() MUST be called once per chunk, then indexed. Calling it
    # inside the FEATURE_NAMES comprehension recomputes the whole feature dict
    # once per feature (~180x) and blows up serve latency.
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.array(rows, dtype=float)


def _predict_all(model, X):
    """Soft-vote the three members under a HARD 1-thread limit.

    CRITICAL -- MEASURED, do not remove. HistGradientBoosting is OpenMP-parallel and
    IGNORES n_jobs entirely, so `_pin_single_thread` does nothing for it. On this
    (shared, 8-miners-per-box) host its predict was measured at:

        OMP_NUM_THREADS=1  -> 0.094 s / 100 chunks
        OMP_NUM_THREADS=16 -> 69.7  s / 100 chunks     (740x slower -- OMP spin-wait)

    Setting os.environ at import time is NOT reliable (bittensor imports numpy long
    before this module, and OpenMP reads the var at library load). threadpool_limits
    clamps at CALL time, which is the only robust fix. A serve timeout costs the whole
    round -- and a live UID that misses a round is scored 0, not excluded.
    """
    def _run():
        p = np.zeros(len(X), dtype=float)
        wsum = 0.0
        for _n, m, w in model["members"]:
            p += w * m.predict_proba(X)[:, 1]
            wsum += w
        return p / wsum
    if threadpool_limits is None:
        return _run()
    with threadpool_limits(limits=1):
        return _run()


def _vote(model, chunks):
    return _predict_all(model, _rows(chunks))


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


_T_HI = 0.00040000000000000002
_T_LO = -0.00040000000000000002


def _decision(model, v):
    """Deployed decision layer, verbatim (strictly monotone, tie-free FLOOR/CAP)."""
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
    d = _T_HI - t[top].min()
    if d > 0.0:
        t[top] = t[top] + d
    if cap and rest.size:
        d = t[rest].max() - _T_LO
        if d > 0.0:
            t[rest] = t[rest] - d
    scores = 1.0 / (1.0 + np.exp(-t))
    return [round(float(s), 9) for s in scores]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, _rank01(_vote(m, chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback; score_batch is the real entry (needs batch context)."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        return round(float(_vote(m, [chunk])[0]), 6)
    except Exception:
        return 0.5
