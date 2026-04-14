"""
Finite-K Bayesian switching VAR(1) / AR-HMM with simple conjugate updates.

This module focuses on the Bayesian perspective for a fixed number of modes K.
It is a bridge between the purely finite EM-style model in `hmm_baselines.py`
and the nonparametric HDP-AR-HMM in `hdp_arhmm.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike

from .hmm_baselines import _logsumexp


@dataclass
class BayesianSwitchingAR1:
    """
    Bayesian switching AR(1) model with NIW-like priors (simplified).

    We place Gaussian priors on the AR matrices and inverse-Wishart priors
    on noise covariances, but use simple closed-form updates rather than
    full MNIW machinery to keep implementation compact.
    """

    K: int
    D: int
    kappa0: float = 1.0
    nu0: float = 5.0
    S0_scale: float = 1.0

    def __post_init__(self) -> None:
        self.pi = np.full(self.K, 1.0 / self.K)
        self.A = np.full((self.K, self.K), 1.0 / self.K)
        self.F = np.random.randn(self.K, self.D, self.D) * 0.1
        self.Sigma = np.array([np.eye(self.D) for _ in range(self.K)])

        # hyperparameters for Gaussian prior on vec(F_k)
        self.F0 = np.zeros((self.K, self.D, self.D))
        self.Lambda0 = np.eye(self.D * self.D) / self.kappa0
        self.S0 = np.array([self.S0_scale * np.eye(self.D) for _ in range(self.K)])

    def _log_emission_density(self, y: np.ndarray) -> np.ndarray:
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

        log_alpha[0] = log_pi
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

    def _sample_z(self, y: np.ndarray) -> np.ndarray:
        """
        Block-sample the latent mode sequence z_{1:T} using forward-filter backward-sample.
        """
        T = y.shape[0]
        log_B = self._log_emission_density(y)
        log_pi = np.log(self.pi + 1e-32)
        log_A = np.log(self.A + 1e-32)

        log_alpha = np.zeros((T, self.K))
        log_alpha[0] = log_pi
        for t in range(1, T):
            tmp = log_alpha[t - 1][:, None] + log_A
            log_alpha[t] = _logsumexp(tmp, axis=0) + log_B[t]

        z = np.zeros(T, dtype=int)
        # sample z_T
        log_pT = log_alpha[-1] - _logsumexp(log_alpha[-1])
        z[-1] = np.random.choice(self.K, p=np.exp(log_pT))

        for t in reversed(range(T - 1)):
            log_pt = log_alpha[t] + log_A[:, z[t + 1]]
            log_pt -= _logsumexp(log_pt)
            z[t] = np.random.choice(self.K, p=np.exp(log_pt))
        return z

    def _update_transitions(self, z: np.ndarray, alpha_dir: float = 1.0) -> None:
        K = self.K
        counts = np.zeros((K, K))
        for t in range(len(z) - 1):
            counts[z[t], z[t + 1]] += 1
        # symmetric Dirichlet prior
        A_post = counts + alpha_dir
        self.A = np.stack(
            [np.random.dirichlet(np.maximum(A_post[j], 1e-8).astype(float)) for j in range(K)]
        )

        pi_counts = np.zeros(K)
        pi_counts[z[0]] += 1
        self.pi = np.random.dirichlet(pi_counts + alpha_dir)

    def _update_dynamics(self, y: np.ndarray, z: np.ndarray) -> None:
        Y_prev = y[:-1]
        Y_curr = y[1:]
        for k in range(self.K):
            idx = np.where(z[1:] == k)[0]
            if idx.size == 0:
                continue
            X = Y_prev[idx]
            Y = Y_curr[idx]
            # posterior precision for vec(F_k): Lambda0 + X^T X \kron Sigma^{-1}
            # For simplicity, we use MAP estimate via ridge regression row-wise.
            XtX = X.T @ X
            lam = 1e-2
            try:
                inv = np.linalg.inv(XtX + lam * np.eye(self.D))
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(XtX + lam * np.eye(self.D))
            Fk = (inv @ X.T @ Y).T
            self.F[k] = Fk

            resid = Y - (self.F[k] @ X.T).T
            S_post = resid.T @ resid + self.S0[k]
            nu_post = self.nu0 + X.shape[0]
            # inverse-Wishart mode
            self.Sigma[k] = S_post / (nu_post + self.D + 1)

    def gibbs(self, y: ArrayLike, n_iters: int = 200) -> dict:
        """
        Run a simple Gibbs sampler over latent states and dynamics.

        Returns
        -------
        history : dict
            Stores a few traces for analysis (log-likelihoods, etc.).
        """
        y = np.asarray(y)
        T = y.shape[0]
        # initialize latent modes randomly
        z = np.random.randint(self.K, size=T)
        history = {"loglik": []}

        for _ in range(n_iters):
            # update dynamics given z
            self._update_dynamics(y, z)
            # update transitions
            self._update_transitions(z)
            # resample z
            z = self._sample_z(y)
            # log-likelihood under current parameters
            log_B = self._log_emission_density(y)
            log_pi = np.log(self.pi + 1e-32)
            log_A = np.log(self.A + 1e-32)
            log_alpha = np.zeros((T, self.K))
            log_alpha[0] = log_pi
            for t in range(1, T):
                tmp = log_alpha[t - 1][:, None] + log_A
                log_alpha[t] = _logsumexp(tmp, axis=0) + log_B[t]
            history["loglik"].append(_logsumexp(log_alpha[-1]))

        history["last_z"] = z
        return history

