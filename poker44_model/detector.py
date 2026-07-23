"""Poker44 bot detector -- PROVEN ens/vote class retrained fresh (vote fuse).
ET700/RF700/HGB700 over RAW C2-180 behavioral features (train==serve, inference does
NOT re-sanitize) -> soft-vote .45/.25/.30 (VOTE) -> strictly-monotone NOISO decision (FLOOR lifts exactly
ceil(FLOOR*n) chunks over 0.5 -> hard-zero-safe, distinct scores). Selected on LIVE
round history (ALLDATA_0721 panel), not offline proxies. All estimators single-thread.
"""
from __future__ import annotations
import os
import numpy as np, joblib
try:
    from .features import chunk_features, FEATURE_NAMES
except ImportError:
    from features import chunk_features, FEATURE_NAMES
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None
_M = None; _T_HI = 4e-4; _T_LO = -4e-4
def _pin(e):
    try: e.set_params(n_jobs=1)
    except Exception: pass
def _model():
    global _M
    if _M is None:
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        for k in ("et", "rf", "hgb"):
            if k in b: _pin(b[k])
        _M = b
    return _M
def _rank01(s):
    s = np.asarray(s, float)
    return np.zeros_like(s) if s.size <= 1 else np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float)/(s.size-1)
def _rows(chunks):
    out = []
    for c in chunks:
        f = chunk_features(c); out.append([f.get(k, 0.0) for k in FEATURE_NAMES])
    return np.nan_to_num(np.array(out, float))
def _fused(m, chunks):
    X = _rows(chunks)
    def run():
        ps = dict(et=m["et"].predict_proba(X)[:,1], rf=m["rf"].predict_proba(X)[:,1], hgb=m["hgb"].predict_proba(X)[:,1])
        if m.get("fuser") == "rankfuse":
            raw = (_rank01(ps["et"]) + _rank01(ps["rf"]) + _rank01(ps["hgb"])) / 3.0
        else:
            w = m["vote_weights"]
            raw = (w[0]*ps["et"] + w[1]*ps["rf"] + w[2]*ps["hgb"]) / sum(w)
        return _rank01(raw)
    if threadpool_limits is None: return run()
    with threadpool_limits(limits=1): return run()
def _logit(p, eps):
    p = np.clip(np.asarray(p, float), eps, 1-eps); return np.log(p/(1-p))
def _decision(m, fused):
    eps=float(m["EPS"]); q=float(m["Q"]); margin=float(m["MARGIN"]); temp=float(m.get("TEMP",1.0))
    floor=float(m["FLOOR"]); cap=bool(m.get("CAP", False))
    tref=float(m["train_ref_logit"]) - margin; z=_logit(fused, eps)
    if z.size == 0: return []
    t=(z - np.quantile(z, q) + tref)/temp; order=np.argsort(-z, kind="mergesort")
    k=max(1, int(np.ceil(floor*len(t)))); top, rest = order[:k], order[k:]
    d=_T_HI - t[top].min()
    if d > 0: t[top] = t[top] + d
    if cap and rest.size:
        d=t[rest].max() - _T_LO
        if d > 0: t[rest] = t[rest] - d
    return [round(float(s), 9) for s in 1.0/(1.0+np.exp(-t))]
def score_batch(chunks):
    chunks = chunks or []
    if not chunks: return []
    try:
        m=_model(); return _decision(m, _fused(m, chunks))
    except Exception:
        return [0.5]*len(chunks)
def score_chunk(chunk): return 0.5
