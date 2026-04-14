"""
Baseline HMM and AR-HMM implementations.

These are standard finite-K models used as baselines before introducing
nonparametric (HDP) and recurrent extensions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


@dataclass
class GaussianHMM:
    """
    Simple Gaussian HMM with full covariance matrices.

    Parameters
    ----------
    K : int
        Number of hidden states.
    D : int
        Observation dimensionality.
    """

    K: int
    D: int

    def __post_init__(self) -> None:
        self.pi = np.full(self.K, 1.0 / self.K)
        self.A = np.full((self.K, self.K), 1.0 / self.K)
        self.mu = np.random.randn(self.K, self.D)
        self.Sigma = np.array([np.eye(self.D) for _ in range(self.K)])

    # ------------------------------------------------------------------
    # Emission log-likelihood
    # ------------------------------------------------------------------
    def _log_emission_density(self, y: np.ndarray) -> np.ndarray:
        """
        Compute log N(y_t | mu_k, Sigma_k) for all t,k.

        Returns
        -------
        loglik : (T, K) array
        """
        T = y.shape[0]
        loglik = np.empty((T, self.K))
        for k in range(self.K):
            diff = y - self.mu[k]
            try:
                L = np.linalg.cholesky(self.Sigma[k])
            except np.linalg.LinAlgError:
                # regularize
                L = np.linalg.cholesky(self.Sigma[k] + 1e-6 * np.eye(self.D))
            solve = np.linalg.solve(L, diff.T)
            quad = np.sum(solve ** 2, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            loglik[:, k] = -0.5 * (quad + logdet + self.D * np.log(2 * np.pi))
        return loglik

    # ------------------------------------------------------------------
    # Forward-backward (log-space)
    # ------------------------------------------------------------------
    def _forward_backward(self, y: np.ndarray):
        T = y.shape[0]
        log_B = self._log_emission_density(y)

        log_pi = np.log(self.pi + 1e-32)
        log_A = np.log(self.A + 1e-32)

        log_alpha = np.zeros((T, self.K))
        log_beta = np.zeros((T, self.K))

        # forward
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            tmp = log_alpha[t - 1][:, None] + log_A
            log_alpha[t] = _logsumexp(tmp, axis=0) + log_B[t]

        # backward
        log_beta[-1] = 0.0
        for t in reversed(range(T - 1)):
            tmp = log_A + log_B[t + 1] + log_beta[t + 1]
            log_beta[t] = _logsumexp(tmp, axis=1)

        log_gamma = log_alpha + log_beta
        log_Z = _logsumexp(log_gamma, axis=1)
        log_gamma -= log_Z[:, None]
        gamma = np.exp(log_gamma)

        xi = np.zeros((T - 1, self.K, self.K))
        for t in range(T - 1):
            tmp = (
                log_alpha[t][:, None]
                + log_A
                + log_B[t + 1][None, :]
                + log_beta[t + 1][None, :]
            )
            tmp -= _logsumexp(tmp.ravel())
            xi[t] = np.exp(tmp)

        return gamma, xi

    # ------------------------------------------------------------------
    # EM fitting
    # ------------------------------------------------------------------
    def fit(self, y: ArrayLike, n_iters: int = 50) -> None:
        """
        Fit model with EM.

        Parameters
        ----------
        y : array-like, shape (T, D)
        n_iters : int
            Number of EM iterations.
        """
        y = np.asarray(y)
        T = y.shape[0]

        for _ in range(n_iters):
            gamma, xi = self._forward_backward(y)

            # M-step: update pi and A
            self.pi = gamma[0] / np.sum(gamma[0])
            self.A = np.sum(xi, axis=0)
            self.A /= np.sum(self.A, axis=1, keepdims=True)

            # update emission parameters
            for k in range(self.K):
                w = gamma[:, k][:, None]
                w_sum = np.sum(w)
                if w_sum < 1e-8:
                    continue
                mu_k = np.sum(w * y, axis=0) / w_sum
                diff = y - mu_k
                Sigma_k = (w * diff).T @ diff / w_sum
                # small regularization
                Sigma_k += 1e-6 * np.eye(self.D)
                self.mu[k] = mu_k
                self.Sigma[k] = Sigma_k

    def sample(self, T: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample a sequence of length T: (states, observations)."""
        z = np.zeros(T, dtype=int)
        y = np.zeros((T, self.D))

        z[0] = np.random.choice(self.K, p=self.pi)
        y[0] = np.random.multivariate_normal(self.mu[z[0]], self.Sigma[z[0]])
        for t in range(1, T):
            z[t] = np.random.choice(self.K, p=self.A[z[t - 1]])
            y[t] = np.random.multivariate_normal(self.mu[z[t]], self.Sigma[z[t]])
        return z, y


@dataclass
class SwitchingAR1:
    """
    Finite-K switching autoregressive model of order 1 (AR-HMM).

    y_t = A_{z_t} y_{t-1} + e_t,  e_t ~ N(0, Sigma_{z_t})
    """

    K: int
    D: int

    def __post_init__(self) -> None:
        self.pi = np.full(self.K, 1.0 / self.K)
        self.A = np.full((self.K, self.K), 1.0 / self.K)
        self.F = np.random.randn(self.K, self.D, self.D) * 0.1
        self.Sigma = np.array([np.eye(self.D) for _ in range(self.K)])

    def _log_emission_density(self, y: np.ndarray) -> np.ndarray:
        """
        Log-likelihoods p(y_t | y_{t-1}, z_t = k) for t>=1.

        Returns
        -------
        loglik : (T, K), with loglik[0] = 0 since y_0 is treated as given.
        """
        T = y.shape[0]
        loglik = np.zeros((T, self.K))
        for k in range(self.K):
            mean = (self.F[k] @ y[:-1].T).T
            diff = y[1:] - mean
            try:
                L = np.linalg.cholesky(self.Sigma[k])
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(self.Sigma[k] + 1e-6 * np.eye(self.D))
            solve = np.linalg.solve(L, diff.T)
            quad = np.sum(solve ** 2, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            loglik[1:, k] = -0.5 * (quad + logdet + self.D * np.log(2 * np.pi))
        return loglik

    def _forward_backward(self, y: np.ndarray):
        T = y.shape[0]
        log_B = self._log_emission_density(y)

        log_pi = np.log(self.pi + 1e-32)
        log_A = np.log(self.A + 1e-32)

        log_alpha = np.zeros((T, self.K))
        log_beta = np.zeros((T, self.K))

        log_alpha[0] = log_pi  # y0 treated as given
        for t in range(1, T):
            tmp = log_alpha[t - 1][:, None] + log_A
            log_alpha[t] = _logsumexp(tmp, axis=0) + log_B[t]

        log_beta[-1] = 0.0
        for t in reversed(range(T - 1)):
            tmp = log_A + log_B[t + 1] + log_beta[t + 1]
            log_beta[t] = _logsumexp(tmp, axis=1)

        log_gamma = log_alpha + log_beta
        log_Z = _logsumexp(log_gamma, axis=1)
        log_gamma -= log_Z[:, None]
        gamma = np.exp(log_gamma)

        xi = np.zeros((T - 1, self.K, self.K))
        for t in range(T - 1):
            tmp = (
                log_alpha[t][:, None]
                + log_A
                + log_B[t + 1][None, :]
                + log_beta[t + 1][None, :]
            )
            tmp -= _logsumexp(tmp.ravel())
            xi[t] = np.exp(tmp)

        return gamma, xi

    def m_step(self, y: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        T = y.shape[0]

        # update pi and A
        self.pi = gamma[0] / np.sum(gamma[0])
        self.A = np.sum(xi, axis=0)
        self.A /= np.sum(self.A, axis=1, keepdims=True)

        # update AR parameters per mode with weighted least squares
        Y_prev = y[:-1]
        Y_curr = y[1:]
        for k in range(self.K):
            w = gamma[1:, k][:, None]
            w_sum = np.sum(w)
            if w_sum < 1e-8:
                continue
            Xw = (w * Y_prev).T  # D x (T-1)
            Yw = (w * Y_curr).T  # D x (T-1)
            XtX = Xw @ Y_prev
            XtY = Xw @ Y_curr
            # solve F_k * XtX = XtY (row-wise)
            try:
                F_k = np.linalg.solve(XtX + 1e-6 * np.eye(self.D), XtY).T
            except np.linalg.LinAlgError:
                F_k = np.linalg.lstsq(XtX + 1e-6 * np.eye(self.D), XtY, rcond=None)[0].T
            self.F[k] = F_k

            resid = Y_curr - (self.F[k] @ Y_prev.T).T
            Sigma_k = (w * resid).T @ resid / w_sum
            Sigma_k += 1e-6 * np.eye(self.D)
            self.Sigma[k] = Sigma_k

    def fit(self, y: ArrayLike, n_iters: int = 50) -> None:
        """EM-style fitting."""
        y = np.asarray(y)
        for _ in range(n_iters):
            gamma, xi = self._forward_backward(y)
            self.m_step(y, gamma, xi)

    def sample(self, T: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample states and observations."""
        z = np.zeros(T, dtype=int)
        y = np.zeros((T, self.D))
        # initial y0 from standard normal
        y[0] = np.random.randn(self.D)
        z[0] = np.random.choice(self.K, p=self.pi)
        for t in range(1, T):
            z[t] = np.random.choice(self.K, p=self.A[z[t - 1]])
            mean = self.F[z[t]] @ y[t - 1]
            y[t] = np.random.multivariate_normal(mean, self.Sigma[z[t]])
        return z, y

