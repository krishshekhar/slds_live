"""
Evaluation utilities for switching dynamical systems.

Provides:
- Hamming distance between two state sequences (up to label permutation)
- Run-length distribution summaries
- Simple predictive log-likelihood computation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from itertools import permutations


def hamming_distance_perm(z_true: ArrayLike, z_est: ArrayLike, K: int | None = None) -> float:
    """
    Hamming distance between two label sequences, optimized over all
    permutations of the estimated labels (to account for label swapping).
    """
    z_true = np.asarray(z_true, dtype=int)
    z_est = np.asarray(z_est, dtype=int)
    assert z_true.shape == z_est.shape

    if K is None:
        K = max(z_true.max(), z_est.max()) + 1

    best = z_true.size
    for perm in permutations(range(K)):
        mapping = np.array(perm)
        z_perm = mapping[z_est]
        dist = np.sum(z_perm != z_true)
        if dist < best:
            best = dist
    return best / z_true.size


def run_lengths(z: ArrayLike) -> np.ndarray:
    """
    Compute run-lengths for a discrete state sequence.
    """
    z = np.asarray(z, dtype=int)
    if z.size == 0:
        return np.array([])
    lengths = []
    current = z[0]
    length = 1
    for i in range(1, z.size):
        if z[i] == current:
            length += 1
        else:
            lengths.append(length)
            current = z[i]
            length = 1
    lengths.append(length)
    return np.array(lengths, dtype=int)


def empirical_runlength_stats(z: ArrayLike) -> dict:
    """
    Summary statistics for run-lengths.
    """
    rl = run_lengths(z)
    if rl.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0}
    return {"mean": float(rl.mean()), "std": float(rl.std()), "max": int(rl.max())}

