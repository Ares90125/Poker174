"""Poker44 model package — bot detector over 180 sanitization-invariant features."""
from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
