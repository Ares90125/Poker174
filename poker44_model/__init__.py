"""Participant-owned model package for the Poker44 miner (uid7) — v5 sani fix.

Bot detector = ExtraTrees + HistGradientBoosting soft-vote ensemble over the v3
behavioral feature set with fragile identity / raw-magnitude aggregates removed
(candidate C2). Trained on benchmark hands passed through the validator's
prepare_hand_for_miner so the training distribution matches the sanitized live
feed (train==serve); scored by within-batch ranking. Inference does NOT
re-sanitize (live hands are already sanitized validator-side). See detector.py
(inference), features.py (extraction + FEATURE_NAMES), train_model.py (training),
model.joblib (trained model).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
