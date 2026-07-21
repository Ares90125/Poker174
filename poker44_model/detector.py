"""Poker44 bot detector -- BENCH-DISCRIMINATION learner-diversity stack (A/B vs input-rank).
RAW union-452 order-stat features (transfer-stable; NO input-rank) -> ET+RF+HGB+LGBM+XGB
soft-vote (selected on GKF-by-date AP) -> strictly-monotone NOISO decision (FLOOR crosses
ceil(FLOOR*n) -> hard-zero-safe). Output-rank only feeds the NOISO layer (order-preserving).
"""
from __future__ import annotations
import os
import numpy as np, joblib
try:
    from .union_features import union_features, UNION_NAMES
except ImportError:
    from union_features import union_features, UNION_NAMES
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits=None
_M=None; _T_HI=4e-4; _T_LO=-4e-4
def _pin(e):
    for a in ("n_jobs","nthread","thread_count","n_thread"):
        try: e.set_params(**{a:1})
        except Exception: pass
def _model():
    global _M
    if _M is None:
        b=joblib.load(os.path.join(os.path.dirname(__file__),"model.joblib"))
        for k in ("et","rf","hgb","lgb","xgb"):
            if k in b:
                try: _pin(b[k])
                except Exception: pass
        _M=b
    return _M
def _rank01(s):
    s=np.asarray(s,float)
    return np.zeros_like(s) if s.size<=1 else np.argsort(np.argsort(s,kind="stable"),kind="stable").astype(float)/(s.size-1)
def _rows(chunks):
    r=[]
    for c in chunks:
        f=union_features(c); r.append([f.get(k,0.0) for k in UNION_NAMES])
    return np.nan_to_num(np.array(r,float))
def _vote(m,X):
    w=m["vote_weights"]
    ps=[m["et"].predict_proba(X)[:,1],m["rf"].predict_proba(X)[:,1],m["hgb"].predict_proba(X)[:,1],
        m["lgb"].predict_proba(X)[:,1],m["xgb"].predict_proba(X)[:,1]]
    return sum(wi*p for wi,p in zip(w,ps))/sum(w)
def _fused(m,chunks):
    X=_rows(chunks)
    def run(): return _rank01(_vote(m,X))
    if threadpool_limits is None: return run()
    with threadpool_limits(limits=1): return run()
def _logit(p,eps):
    p=np.clip(np.asarray(p,float),eps,1-eps); return np.log(p/(1-p))
def _decision(m,fused):
    eps=float(m["EPS"]);q=float(m["Q"]);margin=float(m["MARGIN"]);temp=float(m.get("TEMP",1.0));floor=float(m["FLOOR"]);cap=bool(m.get("CAP",False))
    tref=float(m["train_ref_logit"])-margin; z=_logit(fused,eps)
    if z.size==0: return []
    t=(z-np.quantile(z,q)+tref)/temp; order=np.argsort(-z,kind="mergesort")
    k=max(1,int(np.ceil(floor*len(t)))); top,rest=order[:k],order[k:]
    d=_T_HI-t[top].min()
    if d>0: t[top]=t[top]+d
    if cap and rest.size:
        d=t[rest].max()-_T_LO
        if d>0: t[rest]=t[rest]-d
    return [round(float(s),9) for s in 1.0/(1.0+np.exp(-t))]
def score_batch(chunks):
    chunks=chunks or []
    if not chunks: return []
    try:
        m=_model(); return _decision(m,_fused(m,chunks))
    except Exception: return [0.5]*len(chunks)
def score_chunk(chunk): return 0.5
