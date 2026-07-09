#!/usr/bin/env python3
"""Reproduce this fork's poker44_model/model.joblib from the PUBLIC benchmark.

This script is DOCUMENTATION of the exact training flow. It is NOT served, NOT
loaded at inference time, and NOT listed in the miner's implementation_files /
implementation_sha256 (the served artifact is model.joblib + detector.py +
features.py + mlp_bag.py + mlp_member.py). It exists so a reviewer can rebuild
model.joblib end-to-end using ONLY this repo's own code plus the public data.

WHAT THE MODEL IS
    * Estimator: a 5-member bag of standardized Torch MLPs (poker44_model.mlp_bag
      .BagMLP over poker44_model.mlp_member.TorchMLPClassifier). Each member
      standardizes inputs on the train mean/std and early-stops on validation
      LOSS, so the bag EXTRAPOLATES a discriminative ordering onto the validator-
      sanitized (out-of-distribution) live feed instead of collapsing flat.
    * Inputs: C2's 180-dim feature vector (poker44_model.features
      .chunk_features -> the FEATURE_NAMES-ordered row) computed over each hand
      chunk. These are sanitization-invariant behavioral / entropy / duplication-
      signature aggregates (shares, entropy, run-lengths, bb-normalized amounts);
      raw-magnitude / identity columns are deliberately excluded so the row is
      the SAME whether computed on a raw or a validator-sanitized chunk.
    * Served output: NOT this bag's raw probability. detector.py wraps it in a
      rank-anchored calibrated logistic (within each served batch, map the bag
      probability to its within-batch rank u in [0,1], then
      sigmoid(TEMP*(u-(1-BOT_FRACTION)))) so the top BOT_FRACTION of each batch
      crosses 0.5. That transform is rank-preserving, so AP / bot-recall are
      unchanged vs this bag's ordering; it only shapes the level for the
      validator's new-reward threshold block. See detector.py for the tuning.

TRAINED ONLY ON THE PUBLIC BENCHMARK
    The training matrix is built exclusively from the public hand-chunk benchmark
    (train/raw/chunks_*.json, fetched by train/download.py from the public API at
    api.poker44.net -- no auth). The label y is the benchmark's own groundTruth
    (1 = bot, 0 = human). NO validator-only labels, NO private data, NO live
    captures are used to fit the model.

TRAIN == SERVE (why we sanitize the training data)
    The validator never shows miners raw hands: it projects every hand through
    poker44.validator.payload_view.prepare_hand_for_miner (strips labels/identity,
    re-aliases seats, coarsens bet sizes into bb buckets, samples a deterministic
    action window). To train on the same distribution we serve on, this script
    applies that EXACT function to every benchmark hand before featurizing. So the
    features the model learns on are the features it will see live. (Inference in
    detector.py does NOT re-sanitize -- live chunks already arrive sanitized.)

REPRODUCIBILITY NOTE
    BagMLP derives its member seeds internally as (BASE_SEED*100 + i + 1) for
    i in 0..N_MEMBERS-1 and bags them. Re-running this script reproduces the
    RECIPE -- same estimator class, architecture, feature set, seed policy, and
    public training data -- not necessarily bit-identical weights, because (a) the
    public benchmark grows over time, so a later run trains on more chunks than the
    shipped artifact did, and (b) Torch CPU training has minor platform/threading
    nondeterminism. The resulting model is equivalent by construction.

HOW TO RUN (from the repo root, with the benchmark downloaded)
    python train/download.py                 # populate train/raw/ from the public API
    POKER44_TORCH_THREADS=1 python -m poker44_model.train_model
    #   ...or:  python poker44_model/train_model.py
    # Writes poker44_model/model.joblib. Override inputs/outputs if needed:
    #   POKER44_TRAIN_RAW=/path/to/raw   POKER44_MODEL_OUT=/tmp/model.joblib
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

# =============================== PER-FORK CONFIG ===============================
# Fleet tag: bagB
BASE_SEED = 1            # BagMLP derives member seeds = BASE_SEED*100 + i + 1
HIDDEN_SIZES = (512, 256, 128)      # MLP hidden-layer widths (per member)
DROPOUT = 0.3             # dropout between hidden layers
N_MEMBERS = 5             # bag size
# ==============================================================================

# repo root = parent of poker44_model/ ; makes `poker44` / `poker44_model` importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Bound CPU threads (keeps training deterministic-ish and avoids Torch deadlocks).
try:
    import torch

    torch.set_num_threads(int(os.environ.get("POKER44_TORCH_THREADS", "1")))
except Exception:
    pass

# This fork's OWN code -- no dependency on any private sweep/scratchpad library.
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44_model.features import FEATURE_NAMES, chunk_features
from poker44_model.mlp_bag import BagMLP
import joblib

RAW_DIR = os.environ.get("POKER44_TRAIN_RAW") or os.path.join(REPO_ROOT, "train", "raw")
OUT_PATH = os.environ.get("POKER44_MODEL_OUT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model.joblib"
)


def _sanitize_group(group):
    """Project a benchmark chunk through the validator's per-hand serve view."""
    out = []
    for hand in group or []:
        try:
            out.append(prepare_hand_for_miner(hand))
        except Exception:
            out.append(hand)  # never drop a hand; fall back to the raw payload
    return out


def _feature_row(sanitized_group):
    """FEATURE_NAMES-ordered 180-vector for one sanitized chunk."""
    feats = chunk_features(sanitized_group)
    return [feats.get(k, 0.0) for k in FEATURE_NAMES]


def load_benchmark(raw_dir=RAW_DIR):
    """Load every labeled chunk group from the public benchmark files.

    Returns (groups, labels): groups is a list of chunks (each a list of hand
    dicts), labels is the parallel 0/1 groundTruth (1 = bot, 0 = human).
    """
    files = sorted(glob.glob(os.path.join(raw_dir, "chunks_*.json")))
    if not files:
        raise SystemExit(
            "No benchmark files in %s -- run `python train/download.py` first." % raw_dir
        )
    groups, labels = [], []
    for path in files:
        with open(path) as fh:
            data = json.load(fh)
        for record in data.get("chunks", []):
            chunks = record.get("chunks") or []
            gt = record.get("groundTruth") or []
            for group, label in zip(chunks, gt):
                groups.append(group)
                labels.append(int(label))
    return groups, labels


def build_matrix(groups):
    """Sanitize (train==serve) then featurize every group -> X[n, 180]."""
    X = np.array([_feature_row(_sanitize_group(g)) for g in groups], dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    print(
        "[config] fleet=bagB arch=%s dropout=%s base_seed=%d n_members=%d"
        % (HIDDEN_SIZES, DROPOUT, BASE_SEED, N_MEMBERS),
        flush=True,
    )
    groups, labels = load_benchmark()
    y = np.asarray(labels, dtype=int)
    print(
        "[data] %d groups  bot=%d human=%d  from %s"
        % (len(groups), int((y == 1).sum()), int((y == 0).sum()), RAW_DIR),
        flush=True,
    )

    t0 = time.time()
    X = build_matrix(groups)
    print("[features] X=%s  built in %.0fs" % (str(X.shape), time.time() - t0), flush=True)

    t1 = time.time()
    bag = BagMLP(
        seed=BASE_SEED, n_members=N_MEMBERS, hidden_sizes=HIDDEN_SIZES, dropout=DROPOUT
    ).fit(X, y)
    print(
        "[train] %d-member bag trained in %.0fs  member_seeds=%s"
        % (len(bag.members), time.time() - t1, [c.seed for c in bag.members]),
        flush=True,
    )

    joblib.dump(bag, OUT_PATH)
    print("[saved] %s  (%d bytes)" % (OUT_PATH, os.path.getsize(OUT_PATH)), flush=True)


if __name__ == "__main__":
    main()
