"""
Sticky HDP-AR-HMM with weak-limit truncation and MNIW dynamics updates.

This module implements the HDP-AR-HMM side of Fox et al. (2008):
- weak-limit approximation for HDP transition weights,
- sticky transition bias kappa for persistent regimes,
- blocked latent-state sampling via forward-filter backward-sample,
- matrix-normal inverse-Wishart posterior sampling for VAR dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import invwishart

from .hmm_baselines import _logsumexp


def _lag_design(y: np.ndarray, ar_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Build VAR(r) regression matrices: Y = X B + eps."""
    T, D = y.shape
    n = T - ar_order
    X = np.zeros((n, D * ar_order))
    Y = y[ar_order:]
    for i in range(n):
        t = ar_order + i
        X[i] = np.concatenate([y[t - lag] for lag in range(1, ar_order + 1)], axis=0)
    return X, Y


@dataclass
class StickyHDPARHMM:
    """
    Sticky HDP-AR-HMM with VAR(r) emissions and blocked Gibbs inference.
    """

    L: int
    D: int
    ar_order: int = 1
    alpha: float = 5.0
    gamma: float = 1.0
    kappa: float = 10.0
    prior_scale: float = 1.0
    prior_dof: float | None = None
    prior_reg: float = 1.0
    random_state: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        self.p = self.D * self.ar_order
        self.beta = self.rng.dirichlet(np.full(self.L, self.gamma / self.L))
        self.pi = np.vstack([self._row_prior(j) for j in range(self.L)])

        # B_k is regression matrix of shape (p, D): Y = X B_k + eps
        self.B = self.rng.normal(scale=0.05, size=(self.L, self.p, self.D))
        self.Sigma = np.array([np.eye(self.D) for _ in range(self.L)])

        # MNIW hyperparameters
        self.M0 = np.zeros((self.p, self.D))
        self.V0_inv = self.prior_reg * np.eye(self.p)
        if self.prior_dof is None:
            self.prior_dof = self.D + 4.0
        self.S0 = self.prior_scale * np.eye(self.D)

    def _row_prior(self, j: int) -> np.ndarray:
        base = self.alpha * self.beta
        base[j] += self.kappa
        return base

    def _log_emission_density(self, y: np.ndarray) -> np.ndarray:
        T = y.shape[0]
        X, Y = _lag_design(y, self.ar_order)
        loglik = np.zeros((T, self.L))
        for k in range(self.L):
            mean = X @ self.B[k]
            diff = Y - mean
            try:
                L_chol = np.linalg.cholesky(self.Sigma[k])
            except np.linalg.LinAlgError:
                L_chol = np.linalg.cholesky(self.Sigma[k] + 1e-6 * np.eye(self.D))
            solve = np.linalg.solve(L_chol, diff.T)
            quad = np.sum(solve ** 2, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(L_chol)))
            loglik[self.ar_order :, k] = -0.5 * (
                quad + logdet + self.D * np.log(2.0 * np.pi)
            )
        return loglik

    def _sample_z(self, y: np.ndarray) -> np.ndarray:
        T = y.shape[0]
        log_B = self._log_emission_density(y)
        log_pi = np.log(self.pi + 1e-32)
        log_pi0 = np.log(self.beta + 1e-32)

        log_alpha = np.zeros((T, self.L))
        log_alpha[0] = log_pi0
        for t in range(1, T):
            tmp = log_alpha[t - 1][:, None] + log_pi
            log_alpha[t] = _logsumexp(tmp, axis=0) + log_B[t]

        z = np.zeros(T, dtype=int)
        pT = np.exp(log_alpha[-1] - _logsumexp(log_alpha[-1]))
        z[-1] = self.rng.choice(self.L, p=pT)
        for t in range(T - 2, -1, -1):
            lp = log_alpha[t] + log_pi[:, z[t + 1]]
            pt = np.exp(lp - _logsumexp(lp))
            z[t] = self.rng.choice(self.L, p=pt)
        return z

    def _sample_transitions(self, z: np.ndarray) -> None:
        N = np.zeros((self.L, self.L))
        for t in range(len(z) - 1):
            N[z[t], z[t + 1]] += 1.0

        # weak-limit global weights
        m = N.sum(axis=0)
        self.beta = self.rng.dirichlet(m + self.gamma / self.L)

        # sticky transition rows
        for j in range(self.L):
            self.pi[j] = self.rng.dirichlet(self._row_prior(j) + N[j])

    def _sample_matrix_normal(
        self, M: np.ndarray, V: np.ndarray, Sigma: np.ndarray
    ) -> np.ndarray:
        """Sample B ~ MN(M, V, Sigma) using vec covariance Sigma kron V."""
        cov = np.kron(Sigma, V)
        mean = M.reshape(-1, order="F")
        vec = self.rng.multivariate_normal(mean=mean, cov=cov)
        return vec.reshape(M.shape, order="F")

    def _sample_dynamics(self, y: np.ndarray, z: np.ndarray) -> None:
        X, Y = _lag_design(y, self.ar_order)
        z_eff = z[self.ar_order :]
        for k in range(self.L):
            idx = np.where(z_eff == k)[0]
            if idx.size == 0:
                self.B[k] = self._sample_matrix_normal(
                    self.M0, np.linalg.inv(self.V0_inv), self.Sigma[k]
                )
                self.Sigma[k] = invwishart.rvs(df=self.prior_dof, scale=self.S0)
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
            nun = self.prior_dof + idx.size
            self.Sigma[k] = invwishart.rvs(df=nun, scale=Sn)
            self.B[k] = self._sample_matrix_normal(Mn, Vn, self.Sigma[k])

    def gibbs(
        self,
        y: ArrayLike,
        n_iters: int = 500,
        burn_in: int = 100,
        *,
        z_init: np.ndarray | None = None,
    ) -> dict:
        """
        Run blocked Gibbs sampling for sticky HDP-AR-HMM.
        """
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        T = y.shape[0]
        if T <= self.ar_order:
            raise ValueError("Time series too short for specified ar_order.")

        if z_init is not None:
            z = np.asarray(z_init, dtype=int).copy()
            if z.shape != (T,):
                raise ValueError(f"z_init must have shape ({T},), got {z.shape}.")
            z = np.clip(z, 0, self.L - 1)
        else:
            z = self.rng.integers(0, self.L, size=T)
        history: dict[str, list] = {"loglik": [], "num_active": [], "z_samples": []}

        for it in range(n_iters):
            self._sample_dynamics(y, z)
            self._sample_transitions(z)
            z = self._sample_z(y)

            log_B = self._log_emission_density(y)
            log_pi = np.log(self.pi + 1e-32)
            log_alpha = np.zeros((T, self.L))
            log_alpha[0] = np.log(self.beta + 1e-32)
            for t in range(1, T):
                log_alpha[t] = _logsumexp(log_alpha[t - 1][:, None] + log_pi, axis=0) + log_B[t]
            history["loglik"].append(float(_logsumexp(log_alpha[-1])))
            history["num_active"].append(int(np.unique(z[self.ar_order :]).size))
            if it >= burn_in:
                history["z_samples"].append(z.copy())

        history["last_z"] = z
        return history

