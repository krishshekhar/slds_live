"""
Recurrent AR-HMM (rAR-HMM) with stick-breaking transitions and Polya-Gamma Gibbs.

This implements the observed-state special case of recurrent SLDS from
Linderman et al. (2017), using:
- stick-breaking logistic transition probabilities,
- Polya-Gamma augmentation for Bayesian recurrence updates,
- MNIW updates for mode-specific AR dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import invwishart

from .hmm_baselines import _logsumexp


def _lag_design(y: np.ndarray, ar_order: int) -> tuple[np.ndarray, np.ndarray]:
    T, D = y.shape
    n = T - ar_order
    X = np.zeros((n, D * ar_order))
    Y = y[ar_order:]
    for i in range(n):
        t = ar_order + i
        X[i] = np.concatenate([y[t - lag] for lag in range(1, ar_order + 1)], axis=0)
    return X, Y


def _sample_pg1(c: np.ndarray, trunc: int, rng: np.random.Generator) -> np.ndarray:
    """
    Approximate samples from PG(1, c) via truncated infinite sum.
    """
    c = np.asarray(c, dtype=float)
    if c.ndim == 0:
        c = c[None]
    n = np.arange(1, trunc + 1, dtype=float)
    denom_base = (n - 0.5) ** 2
    out = np.zeros_like(c)
    for i, ci in enumerate(c):
        gam = rng.gamma(shape=1.0, scale=1.0, size=trunc)
        denom = denom_base + (ci * ci) / (4.0 * np.pi * np.pi)
        out[i] = np.sum(gam / denom) / (2.0 * np.pi * np.pi)
    return out


def _stick_breaking_probs(nu: np.ndarray) -> np.ndarray:
    """
    nu shape (..., K-1) -> probs shape (..., K)
    """
    sig = 1.0 / (1.0 + np.exp(-nu))
    K1 = nu.shape[-1]
    K = K1 + 1
    out = np.zeros(nu.shape[:-1] + (K,))
    rem = np.ones(nu.shape[:-1])
    for k in range(K1):
        out[..., k] = rem * sig[..., k]
        rem = rem * (1.0 - sig[..., k])
    out[..., -1] = rem
    return out


@dataclass
class RecurrentARHMMPG:
    K: int
    D: int
    ar_order: int = 1
    pg_trunc: int = 200
    transition_prior_var: float = 10.0
    dynamics_prior_scale: float = 1.0
    dynamics_prior_dof: float | None = None
    dynamics_prior_reg: float = 1.0
    random_state: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        self.p = self.D * self.ar_order
        self.feature_dim = self.p + 1  # include bias

        self.B = self.rng.normal(scale=0.05, size=(self.K, self.p, self.D))
        self.Sigma = np.array([np.eye(self.D) for _ in range(self.K)])

        # Stick-breaking recurrence weights, conditioned on previous state.
        self.W = self.rng.normal(scale=0.1, size=(self.K, self.K - 1, self.feature_dim))
        self.pi0 = np.full(self.K, 1.0 / self.K)

        if self.dynamics_prior_dof is None:
            self.dynamics_prior_dof = self.D + 4.0
        self.M0 = np.zeros((self.p, self.D))
        self.V0_inv = self.dynamics_prior_reg * np.eye(self.p)
        self.S0 = self.dynamics_prior_scale * np.eye(self.D)

    def _log_emission_density(self, y: np.ndarray) -> np.ndarray:
        T = y.shape[0]
        X, Y = _lag_design(y, self.ar_order)
        out = np.zeros((T, self.K))
        for k in range(self.K):
            mean = X @ self.B[k]
            diff = Y - mean
            try:
                L = np.linalg.cholesky(self.Sigma[k])
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(self.Sigma[k] + 1e-6 * np.eye(self.D))
            solve = np.linalg.solve(L, diff.T)
            quad = np.sum(solve ** 2, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            out[self.ar_order :, k] = -0.5 * (quad + logdet + self.D * np.log(2.0 * np.pi))
        return out

    def _transition_log_probs(self, y: np.ndarray) -> np.ndarray:
        X, _ = _lag_design(y, self.ar_order)
        Xf = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        logP = np.zeros((X.shape[0], self.K, self.K))
        for j in range(self.K):
            nu = Xf @ self.W[j].T
            probs = _stick_breaking_probs(nu)
            logP[:, j, :] = np.log(probs + 1e-32)
        return logP

    def _sample_z(self, y: np.ndarray) -> np.ndarray:
        T = y.shape[0]
        log_B = self._log_emission_density(y)
        log_trans = self._transition_log_probs(y)

        log_alpha = np.full((T, self.K), -np.inf)
        log_alpha[: self.ar_order] = np.log(self.pi0 + 1e-32)
        for t in range(self.ar_order, T):
            if t == self.ar_order:
                prev = np.log(self.pi0 + 1e-32)
            else:
                prev = _logsumexp(log_alpha[t - 1][:, None] + log_trans[t - self.ar_order - 1], axis=0)
            log_alpha[t] = prev + log_B[t]

        z = np.zeros(T, dtype=int)
        pT = np.exp(log_alpha[-1] - _logsumexp(log_alpha[-1]))
        z[-1] = self.rng.choice(self.K, p=pT)
        for t in range(T - 2, self.ar_order - 1, -1):
            lp = log_alpha[t] + log_trans[t - self.ar_order, :, z[t + 1]]
            pt = np.exp(lp - _logsumexp(lp))
            z[t] = self.rng.choice(self.K, p=pt)
        z[: self.ar_order] = z[self.ar_order]
        return z

    def _sample_matrix_normal(
        self, M: np.ndarray, V: np.ndarray, Sigma: np.ndarray
    ) -> np.ndarray:
        cov = np.kron(Sigma, V)
        vec = self.rng.multivariate_normal(M.reshape(-1, order="F"), cov)
        return vec.reshape(M.shape, order="F")

    def _sample_dynamics(self, y: np.ndarray, z: np.ndarray) -> None:
        X, Y = _lag_design(y, self.ar_order)
        z_eff = z[self.ar_order :]
        for k in range(self.K):
            idx = np.where(z_eff == k)[0]
            if idx.size == 0:
                self.Sigma[k] = invwishart.rvs(df=self.dynamics_prior_dof, scale=self.S0)
                self.B[k] = self._sample_matrix_normal(
                    self.M0, np.linalg.inv(self.V0_inv), self.Sigma[k]
                )
                continue
            Xk = X[idx]
            Yk = Y[idx]
            Vn_inv = self.V0_inv + Xk.T @ Xk
            Vn = np.linalg.inv(Vn_inv + 1e-12 * np.eye(self.p))
            Mn = Vn @ (self.V0_inv @ self.M0 + Xk.T @ Yk)
            Sn = (
                self.S0
                + Yk.T @ Yk
                + self.M0.T @ self.V0_inv @ self.M0
                - Mn.T @ Vn_inv @ Mn
            )
            Sn = 0.5 * (Sn + Sn.T) + 1e-8 * np.eye(self.D)
            nun = self.dynamics_prior_dof + idx.size
            self.Sigma[k] = invwishart.rvs(df=nun, scale=Sn)
            self.B[k] = self._sample_matrix_normal(Mn, Vn, self.Sigma[k])

    def _sample_recurrence(self, y: np.ndarray, z: np.ndarray) -> None:
        X, _ = _lag_design(y, self.ar_order)
        Xf = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        z_prev = z[self.ar_order - 1 : -1]
        z_next = z[self.ar_order :]
        prior_prec = 1.0 / self.transition_prior_var

        for j in range(self.K):
            mask_j = z_prev == j
            if not np.any(mask_j):
                continue
            Xj = Xf[mask_j]
            yj = z_next[mask_j]

            for k in range(self.K - 1):
                eligible = yj >= k
                if not np.any(eligible):
                    continue
                Xjk = Xj[eligible]
                target = (yj[eligible] == k).astype(float)
                w_old = self.W[j, k]
                c = Xjk @ w_old
                omega = _sample_pg1(c, self.pg_trunc, self.rng)
                kappa = target - 0.5

                XtOmega = Xjk.T * omega[None, :]
                precision = XtOmega @ Xjk + prior_prec * np.eye(self.feature_dim)
                cov = np.linalg.inv(precision + 1e-12 * np.eye(self.feature_dim))
                mean = cov @ (Xjk.T @ kappa)
                self.W[j, k] = self.rng.multivariate_normal(mean, cov)

    def gibbs(self, y: ArrayLike, n_iters: int = 300, burn_in: int = 100) -> dict:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[0] <= self.ar_order:
            raise ValueError("Time series too short for ar_order.")

        z = self.rng.integers(0, self.K, size=y.shape[0])
        history: dict[str, list] = {"loglik": [], "z_samples": []}

        for it in range(n_iters):
            self._sample_dynamics(y, z)
            self._sample_recurrence(y, z)
            z = self._sample_z(y)

            log_B = self._log_emission_density(y)
            log_trans = self._transition_log_probs(y)
            T = y.shape[0]
            log_alpha = np.full((T, self.K), -np.inf)
            log_alpha[: self.ar_order] = np.log(self.pi0 + 1e-32)
            for t in range(self.ar_order, T):
                if t == self.ar_order:
                    prev = np.log(self.pi0 + 1e-32)
                else:
                    prev = _logsumexp(
                        log_alpha[t - 1][:, None] + log_trans[t - self.ar_order - 1], axis=0
                    )
                log_alpha[t] = prev + log_B[t]
            history["loglik"].append(float(_logsumexp(log_alpha[-1])))
            if it >= burn_in:
                history["z_samples"].append(z.copy())

        history["last_z"] = z
        return history

