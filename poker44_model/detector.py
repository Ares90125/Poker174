"""Poker44 bot detector -- STACK6 (LGBM+XGB+CatBoost+ET+RF+HGB -> LR meta) over
the INPUT-RANK VOTE-family's 452-dim UNION-ORDERSTAT row.

Ports the #1 live miner's (TakashiAkio poker44_ckmon_1) 6-model heterogeneous
stacked-ensemble CONSTRUCTION -- LightGBM + XGBoost + CatBoost + ExtraTrees +
RandomForest + HistGradientBoosting, each an independent base learner, combined
by a trained LogisticRegression meta-head over out-of-fold base probabilities
(honest stacking, GroupKFold-by-date) -- onto OUR transfer-stable feature
surface and decision layer instead of theirs. See
train/dev/families/WHYLOW_0729/RESULT.md for the reward-math case for why
"more decorrelated ensembling" is the one grounded lever, and
train/dev/families/STACK6_0730/RESULT.md for the full port + verification.

Two proven levers stacked on the FEATURE side (unchanged from INPUTRANK_MAX):
  (1) the transfer-stable UNION order-statistic feature surface (features_v2
      order-stats + base_features chunk_features, magnitude cols dropped), and
  (2) the leyr/stack233 INPUT-side rank transform -- every feature mapped to
      its WITHIN-SERVED-WINDOW percentile rank BEFORE the learners.

Decision layer = strictly-monotone NOISO (FLOOR lifts exactly ceil(FLOOR*n)
chunks over 0.5 -> hard-zero-safe), UNCHANGED from INPUTRANK_MAX/detector.py --
only the thing being fed into it changed (vote of 3 -> stacked meta of 6).
Inference does NOT sanitize (validator sanitizes live chunks). All 6 base
learners are pinned to a single thread at load and at every predict call.
"""
from __future__ import annotations
import os
import numpy as np
import joblib
try:
    from .union_features import union_features, UNION_NAMES
except ImportError:
    from union_features import union_features, UNION_NAMES

try:
    import torch  # noqa: F401
    torch.set_num_threads(1)
except Exception:
    pass
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

_MODEL = None
_T_HI = 4e-4
_T_LO = -4e-4
_MODEL_ORDER = ("lgbm", "xgb", "cat", "et", "rf", "hgb")


def _pin_single_thread(est):
    for attr in ("n_jobs", "nthread", "thread_count"):
        try:
            est.set_params(**{attr: 1})
        except Exception:
            # CatBoost refuses set_params on a fitted model -- its thread
            # count is pinned per-call instead, see _base_proba below.
            pass
    for holder in ("estimators_", "estimators"):
        try:
            for sub in getattr(est, holder):
                _pin_single_thread(sub[1] if isinstance(sub, tuple) else sub)
        except Exception:
            pass


def _model():
    global _MODEL
    if _MODEL is None:
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        for key in _MODEL_ORDER:
            if key in b:
                try:
                    _pin_single_thread(b[key])
                except Exception:
                    pass
        _MODEL = b
    return _MODEL


def _rank01(s):
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def _rank01_cols(X):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n <= 1:
        return np.zeros_like(X)
    r = np.argsort(np.argsort(X, axis=0, kind="stable"), axis=0, kind="stable").astype(float)
    return r / (n - 1)


def _rows(chunks):
    rows = []
    for c in chunks:
        f = union_features(c)
        rows.append([f.get(k, 0.0) for k in UNION_NAMES])
    return np.nan_to_num(np.array(rows, dtype=float))


def _base_proba(model, name, Xr):
    """One base learner's P(bot) column, single-thread pinned at call time.

    LightGBM/XGBoost/ExtraTrees/RandomForest respect the n_jobs=1 baked into
    the fitted estimator (set in _pin_single_thread). CatBoost rejects
    set_params on a fitted model, so its thread count is pinned via the
    `thread_count` kwarg accepted directly by predict_proba (verified
    numerically identical to the unpinned prediction -- pinning threads does
    not change the output, only the degree of parallelism).
    HistGradientBoostingClassifier has no n_jobs knob at all; its OpenMP
    threads are bounded by the threadpool_limits(limits=1) context the caller
    wraps every predict call in.
    """
    m = model[name]
    if name == "cat":
        return np.asarray(m.predict_proba(Xr, thread_count=1))[:, 1]
    return np.asarray(m.predict_proba(Xr))[:, 1]


def _stack_prob(model, Xr):
    cols = [_base_proba(model, name, Xr) for name in _MODEL_ORDER]
    Z = np.stack(cols, axis=1)
    return np.asarray(model["meta"].predict_proba(Z))[:, 1]


def _bag_fused(model, chunks):
    X = _rows(chunks)
    Xr = _rank01_cols(X)

    def _run():
        return _rank01(_stack_prob(model, Xr))

    if threadpool_limits is None:
        return _run()
    with threadpool_limits(limits=1):
        return _run()


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _decision(model, fused):
    eps = float(model["EPS"]); q = float(model["Q"]); margin = float(model["MARGIN"])
    temp = float(model.get("TEMP", 1.0)); floor = float(model["FLOOR"]); cap = bool(model.get("CAP", False))
    tref = float(model["train_ref_logit"]) - margin
    z = _logit(fused, eps)
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
    return [round(float(s), 9) for s in 1.0 / (1.0 + np.exp(-t))]


def score_batch(chunks):
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, _bag_fused(m, chunks))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    return 0.5
