"""
Sticky HDP-SLDS with weak-limit truncation and Gibbs sampling.

Implements the key components from Fox et al. (2008) for the SLDS case:
- sticky HDP transition prior (weak-limit approximation),
- latent continuous-state sampling with FFBS in time-varying LDS,
- MNIW posterior sampling for mode-specific linear dynamics,
- inverse-Wishart update for shared observation covariance.

Extensions vs. a minimal collapsed Dirichlet-only sampler:
- **(A)** HDP global weights ``beta`` via Chinese Restaurant Table (CRT) auxiliary
  counts ``m_{jk}`` (Teh et al., 2006; Fox et al., 2008), then
  ``beta | m, gamma ~ Dirichlet(gamma/L + M_k)`` with ``M_k = sum_j m_{jk}``.
- **(B)** Explicit backward link messages ``log m_{t+1,t}(z_t) = log pi[z_t, z_{t+1}]``
  in the discrete backward sampling pass (same math as folding into FFBS).
- **(C)** Optional Escobar--West-style draw for ``alpha`` and random-walk MH for
  ``gamma`` and ``kappa`` (Gamma priors on the positive reals).
- **(D)** After each ``z`` update, **canonical permutation** of labels sorting by
  ``beta`` (descending) to reduce label-switching in the truncated weak limit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln
from scipy.stats import invwishart

from .hmm_baselines import _logsumexp


def _sample_crt(n: int, strength: float, rng: np.random.Generator) -> int:
    """
    Chinese Restaurant Table draw: auxiliary table count for ``n`` customers
    and concentration ``strength`` (e.g. ``alpha * beta_k`` in the HDP row).
    """
    if n <= 0:
        return 0
    strength = max(float(strength), 1e-14)
    ell = np.arange(1, n + 1, dtype=float)
    p = np.clip(strength / (strength + ell - 1.0), 0.0, 1.0)
    return int(rng.binomial(1, p).sum())


def _dirichlet_logpdf_vec(x: np.ndarray, conc: np.ndarray) -> float:
    c = np.asarray(conc, dtype=float)
    x = np.asarray(x, dtype=float)
    if np.any(c <= 0) or np.any(x <= 0):
        return -math.inf
    return float(gammaln(c.sum()) - gammaln(c).sum() + np.sum((c - 1.0) * np.log(x)))


def _log_gaussian(diff: np.ndarray, cov: np.ndarray) -> float:
    D = diff.shape[0]
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(D))
    solve = np.linalg.solve(L, diff)
    quad = float(solve.T @ solve)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (quad + logdet + D * np.log(2.0 * np.pi))


@dataclass
class StickyHDPSLDS:
    L: int
    state_dim: int
    obs_dim: int
    alpha: float = 5.0
    gamma: float = 1.0
    kappa: float = 10.0
    prior_scale: float = 1.0
    prior_dof: float | None = None
    prior_reg: float = 1.0
    obs_prior_scale: float = 1.0
    obs_prior_dof: float | None = None
    random_state: int | None = None
    # --- HDP / identifiability (see module docstring) ---
    use_hdp_auxiliary_beta: bool = True
    canonicalize_labels: bool = True
    sample_alpha: bool = True
    # Joint Metropolis–Hastings on (gamma, beta, pi) | z (valid); expensive but optional.
    sample_gamma_mh: bool = False
    sample_kappa_mh: bool = True
    # Gamma(shape, rate) priors on positive hypers; used only if *_mh / sample_alpha.
    alpha_prior_shape: float = 2.0
    alpha_prior_rate: float = 0.5
    gamma_prior_shape: float = 2.0
    gamma_prior_rate: float = 0.5
    kappa_prior_shape: float = 2.0
    kappa_prior_rate: float = 0.2
    mh_log_step: float = 0.12

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        M = self.state_dim
        D = self.obs_dim
        self.beta = self.rng.dirichlet(np.full(self.L, self.gamma / self.L))
        self.pi = np.vstack([self._transition_prior_row(j) for j in range(self.L)])
        # Last CRT column sums M_k and backward messages (for diagnostics / papers).
        self._last_M_from_crt = np.zeros(self.L)
        self._last_N = np.zeros((self.L, self.L))
        self._last_log_forward: np.ndarray | None = None
        self._last_log_backward_link: np.ndarray | None = None

        self.A = self.rng.normal(scale=0.05, size=(self.L, M, M))
        self.Q = np.array([np.eye(M) for _ in range(self.L)])

        # Following Fox et al., we keep C fixed for identifiability.
        self.C = np.eye(D, M)
        self.R = np.eye(D)

        self.M0 = np.zeros((M, M))  # regression Y = X B, where B is MxM
        self.V0_inv = self.prior_reg * np.eye(M)
        if self.prior_dof is None:
            self.prior_dof = M + 4.0
        self.S0 = self.prior_scale * np.eye(M)

        if self.obs_prior_dof is None:
            self.obs_prior_dof = D + 4.0
        self.R0 = self.obs_prior_scale * np.eye(D)

    def _transition_prior_row(self, j: int) -> np.ndarray:
        base = self.alpha * self.beta
        base[j] += self.kappa
        return base

    def _transition_counts(self, z: np.ndarray) -> np.ndarray:
        N = np.zeros((self.L, self.L))
        for t in range(len(z) - 1):
            N[int(z[t]), int(z[t + 1])] += 1.0
        return N

    def _log_transition_likelihood_pi_beta(
        self, z: np.ndarray, pi: np.ndarray, beta: np.ndarray
    ) -> float:
        T = int(z.size)
        ll = np.log(beta[int(z[0])] + 1e-32)
        for t in range(T - 1):
            ll += np.log(pi[int(z[t]), int(z[t + 1])] + 1e-32)
        return float(ll)

    def _mh_joint_gamma_beta_pi(self, z: np.ndarray, N: np.ndarray) -> None:
        """MH on (gamma, beta, pi) given CRT counts M and transition counts N (symmetric RW on log gamma)."""
        Mvec = self._last_M_from_crt
        I = np.eye(self.L)

        def _log_target(gam: float, bet: np.ndarray, pimat: np.ndarray) -> float:
            conc_b = gam / self.L + Mvec
            if np.any(conc_b <= 0):
                return -math.inf
            lt = self._log_transition_likelihood_pi_beta(z, pimat, bet)
            lt += (self.gamma_prior_shape - 1.0) * np.log(max(gam, 1e-12))
            lt -= self.gamma_prior_rate * gam
            lt += _dirichlet_logpdf_vec(bet, conc_b)
            for j in range(self.L):
                conc = np.maximum(self.alpha * bet + self.kappa * I[j] + N[j], 1e-12)
                lt += _dirichlet_logpdf_vec(pimat[j], conc)
            return float(lt)

        log_cur = _log_target(self.gamma, self.beta, self.pi)
        log_g0 = np.log(max(self.gamma, 1e-12))
        log_g1 = log_g0 + self.rng.normal(0.0, self.mh_log_step)
        gamma_prop = float(np.clip(np.exp(log_g1), 1e-4, 500.0))
        conc_b_prop = gamma_prop / self.L + Mvec
        if np.any(conc_b_prop <= 0):
            return
        beta_prop = self.rng.dirichlet(np.maximum(conc_b_prop, 1e-12))
        pi_prop = np.zeros((self.L, self.L))
        for j in range(self.L):
            pi_prop[j] = self.rng.dirichlet(
                np.maximum(self.alpha * beta_prop + self.kappa * I[j] + N[j], 1e-12)
            )
        log_prop = _log_target(gamma_prop, beta_prop, pi_prop)
        if log_prop - log_cur > np.log(self.rng.random()):
            self.gamma = gamma_prop
            self.beta = beta_prop
            self.pi = pi_prop

    def _mh_sample_kappa(self, N: np.ndarray) -> None:
        """Random-walk Metropolis on log kappa targeting p(kappa | pi, beta, z) with Gamma prior."""
        log_k0 = np.log(max(self.kappa, 1e-12))
        log_k1 = log_k0 + self.rng.normal(0.0, self.mh_log_step)
        k_prop = float(np.clip(np.exp(log_k1), 1e-6, 500.0))
        I = np.eye(self.L)

        def row_lp(kappa_val: float) -> float:
            s = 0.0
            for j in range(self.L):
                conc = np.maximum(self.alpha * self.beta + kappa_val * I[j] + N[j], 1e-12)
                s += _dirichlet_logpdf_vec(self.pi[j], conc)
            return s

        log_rat = row_lp(k_prop) - row_lp(self.kappa)
        log_rat += (self.kappa_prior_shape - 1.0) * (np.log(k_prop) - np.log(self.kappa))
        log_rat -= self.kappa_prior_rate * (k_prop - self.kappa)
        log_rat += np.log(k_prop) - np.log(self.kappa)
        if log_rat > np.log(self.rng.random()):
            self.kappa = k_prop

    def _sample_alpha_escobar_west(self, n_trans: int) -> None:
        """Escobar & West (1995) style draw for DP concentration (approximate for HDP CRT tables)."""
        if not self.sample_alpha:
            return
        K_tables = int(np.sum(self._last_M_from_crt > 0))
        K_tables = max(K_tables, 1)
        a0, b0 = self.alpha_prior_shape, self.alpha_prior_rate
        n = max(n_trans, 1)
        eta = self.rng.beta(self.alpha + 1.0, n)
        r = max(b0 - np.log(max(eta, 1e-300)), 1e-12)
        numer = a0 + K_tables - 1.0
        denom = max(numer + n * r, 1e-12)
        pi_mix = numer / denom
        if self.rng.random() < pi_mix:
            self.alpha = float(self.rng.gamma(a0 + K_tables, scale=1.0 / r))
        else:
            self.alpha = float(self.rng.gamma(a0 + K_tables - 1.0, scale=1.0 / r))
        self.alpha = float(np.clip(self.alpha, 0.05, 200.0))

    def _update_transition_parameters(self, z: np.ndarray) -> None:
        N = self._transition_counts(z)
        self._last_N = N
        L = self.L

        if self.use_hdp_auxiliary_beta:
            M = np.zeros(L)
            for j in range(L):
                for k in range(L):
                    n_jk = int(N[j, k])
                    strength = max(self.alpha * self.beta[k], 1e-14)
                    M[k] += _sample_crt(n_jk, strength, self.rng)
            self._last_M_from_crt = M
        else:
            self._last_M_from_crt = np.zeros(L)

        if self.sample_gamma_mh and self.use_hdp_auxiliary_beta:
            self._mh_joint_gamma_beta_pi(z, N)
        else:
            if self.use_hdp_auxiliary_beta:
                conc_b = np.maximum(self._last_M_from_crt + self.gamma / L, 1e-12)
                self.beta = self.rng.dirichlet(conc_b)
            else:
                col = np.maximum(N.sum(axis=0) + self.gamma / L, 1e-12)
                self.beta = self.rng.dirichlet(col)
            for j in range(L):
                row = np.maximum(self._transition_prior_row(j) + N[j], 1e-12)
                self.pi[j] = self.rng.dirichlet(row)

        if self.sample_kappa_mh:
            self._mh_sample_kappa(N)

    def _canonicalize_by_beta(self, z: np.ndarray) -> np.ndarray:
        """Permute labels so beta is sorted descending (Fox-style weak-limit identifiability)."""
        if not self.canonicalize_labels:
            return z
        perm = np.argsort(-self.beta)
        inv = np.empty(self.L, dtype=int)
        inv[perm] = np.arange(self.L, dtype=int)
        self.beta = self.beta[perm]
        self.pi = self.pi[np.ix_(perm, perm)]
        self.A = self.A[perm]
        self.Q = self.Q[perm]
        return inv[z]

    def _sample_matrix_normal(
        self, M: np.ndarray, V: np.ndarray, Sigma: np.ndarray
    ) -> np.ndarray:
        cov = np.kron(Sigma, V)
        vec = self.rng.multivariate_normal(M.reshape(-1, order="F"), cov)
        return vec.reshape(M.shape, order="F")

    def _sample_transitions(self, z: np.ndarray) -> None:
        """Backward-compatible name: full HDP/sticky block + optional hyper-MH."""
        self._update_transition_parameters(z)
        self._sample_alpha_escobar_west(max(len(z) - 1, 0))

    def _sample_z(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        log_B = np.zeros((T, self.L))
        for t in range(1, T):
            xtm1 = x[t - 1]
            xt = x[t]
            for k in range(self.L):
                mean = self.A[k] @ xtm1
                log_B[t, k] = _log_gaussian(xt - mean, self.Q[k])

        log_A = np.log(self.pi + 1e-32)
        log_pi0 = np.log(self.beta + 1e-32)
        log_alpha = np.zeros((T, self.L))
        log_alpha[0] = log_pi0
        for t in range(1, T):
            log_alpha[t] = _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_B[t]

        self._last_log_forward = log_alpha.copy()

        # Explicit backward link messages m_{t+1,t}(i) = log pi[i, z_{t+1}] (cf. FFBS notation).
        log_m_tp1_t = np.zeros((max(0, T - 1), self.L))
        z = np.zeros(T, dtype=int)
        z[-1] = self.rng.choice(self.L, p=np.exp(log_alpha[-1] - _logsumexp(log_alpha[-1])))
        for t in range(T - 2, -1, -1):
            log_m_row = log_A[:, z[t + 1]]
            log_m_tp1_t[t] = log_m_row
            lp = log_alpha[t] + log_m_row
            z[t] = self.rng.choice(self.L, p=np.exp(lp - _logsumexp(lp)))
        self._last_log_backward_link = log_m_tp1_t
        return z

    def _ffbs_sample_x(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        T = y.shape[0]
        M = self.state_dim
        I = np.eye(M)

        filt_m = np.zeros((T, M))
        filt_P = np.zeros((T, M, M))
        pred_m = np.zeros((T, M))
        pred_P = np.zeros((T, M, M))

        m_prev = np.zeros(M)
        P_prev = np.eye(M)

        for t in range(T):
            if t == 0:
                m_pred = m_prev
                P_pred = P_prev
            else:
                A_t = self.A[z[t]]
                Q_t = self.Q[z[t]]
                m_pred = A_t @ m_prev
                P_pred = A_t @ P_prev @ A_t.T + Q_t
            pred_m[t] = m_pred
            pred_P[t] = P_pred

            S = self.C @ P_pred @ self.C.T + self.R
            K = P_pred @ self.C.T @ np.linalg.inv(S + 1e-10 * np.eye(self.obs_dim))
            innov = y[t] - self.C @ m_pred
            m_f = m_pred + K @ innov
            P_f = (I - K @ self.C) @ P_pred
            P_f = 0.5 * (P_f + P_f.T) + 1e-10 * I

            filt_m[t] = m_f
            filt_P[t] = P_f
            m_prev, P_prev = m_f, P_f

        x = np.zeros((T, M))
        x[-1] = self.rng.multivariate_normal(filt_m[-1], filt_P[-1])
        for t in range(T - 2, -1, -1):
            A_next = self.A[z[t + 1]]
            P_pred_next = pred_P[t + 1]
            J = filt_P[t] @ A_next.T @ np.linalg.inv(P_pred_next + 1e-10 * np.eye(M))
            mean = filt_m[t] + J @ (x[t + 1] - pred_m[t + 1])
            cov = filt_P[t] - J @ P_pred_next @ J.T
            cov = 0.5 * (cov + cov.T) + 1e-10 * np.eye(M)
            x[t] = self.rng.multivariate_normal(mean, cov)
        return x

    def _sample_dynamics(self, x: np.ndarray, z: np.ndarray) -> None:
        for k in range(self.L):
            idx = np.where(z[1:] == k)[0] + 1
            if idx.size == 0:
                self.Q[k] = invwishart.rvs(df=self.prior_dof, scale=self.S0)
                self.A[k] = self._sample_matrix_normal(
                    self.M0, np.linalg.inv(self.V0_inv), self.Q[k]
                )
                continue
            Xk = x[idx - 1]
            Yk = x[idx]
            Vn_inv = self.V0_inv + Xk.T @ Xk
            Vn = np.linalg.inv(Vn_inv + 1e-12 * np.eye(self.state_dim))
            Mn = Vn @ (self.V0_inv @ self.M0 + Xk.T @ Yk)
            Sn = (
                self.S0
                + Yk.T @ Yk
                + self.M0.T @ self.V0_inv @ self.M0
                - Mn.T @ Vn_inv @ Mn
            )
            Sn = 0.5 * (Sn + Sn.T) + 1e-8 * np.eye(self.state_dim)
            nun = self.prior_dof + idx.size
            self.Q[k] = invwishart.rvs(df=nun, scale=Sn)
            Bk = self._sample_matrix_normal(Mn, Vn, self.Q[k])
            self.A[k] = Bk.T

    def _sample_observation_noise(self, y: np.ndarray, x: np.ndarray) -> None:
        resid = y - (self.C @ x.T).T
        SR = resid.T @ resid + self.R0
        dof = self.obs_prior_dof + y.shape[0]
        self.R = invwishart.rvs(df=dof, scale=SR)

    def _complete_data_log_score(self, y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
        """
        Joint log-density log p(y, x | z, theta) for current samples (diagnostic only;
        not comparable to marginal filter log-likelihoods from AR-HMM models).
        """
        ll = 0.0
        T = y.shape[0]
        for t in range(T):
            ll += _log_gaussian(y[t] - self.C @ x[t], self.R)
        for t in range(1, T):
            mean = self.A[z[t]] @ x[t - 1]
            ll += _log_gaussian(x[t] - mean, self.Q[z[t]])
        return float(ll)

    def gibbs(
        self,
        y: ArrayLike,
        n_iters: int = 300,
        burn_in: int = 100,
        *,
        z_init: np.ndarray | None = None,
        x_init: np.ndarray | None = None,
    ) -> dict:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[1] != self.obs_dim:
            raise ValueError("obs_dim does not match data dimension.")

        T = y.shape[0]
        if z_init is not None or x_init is not None:
            if z_init is None or x_init is None:
                raise ValueError("Provide both z_init and x_init, or neither.")
            z = np.asarray(z_init, dtype=int).copy()
            x = np.asarray(x_init, dtype=float).copy()
            if z.shape != (T,):
                raise ValueError(f"z_init must have shape ({T},), got {z.shape}.")
            if x.shape != (T, self.state_dim):
                raise ValueError(
                    f"x_init must have shape ({T}, {self.state_dim}), got {x.shape}."
                )
            z = np.clip(z, 0, self.L - 1)
        else:
            z = self.rng.integers(0, self.L, size=T)
            x = self.rng.normal(size=(T, self.state_dim))
        history: dict[str, list] = {
            "num_active": [],
            "loglik": [],
            "alpha": [],
            "gamma": [],
            "kappa": [],
            "z_samples": [],
            "x_samples": [],
        }

        for it in range(n_iters):
            x = self._ffbs_sample_x(y, z)
            self._sample_dynamics(x, z)
            self._sample_observation_noise(y, x)
            self._sample_transitions(z)
            z = self._sample_z(x)
            z = self._canonicalize_by_beta(z)

            history["loglik"].append(self._complete_data_log_score(y, x, z))
            history["num_active"].append(int(np.unique(z).size))
            history["alpha"].append(float(self.alpha))
            history["gamma"].append(float(self.gamma))
            history["kappa"].append(float(self.kappa))
            if it >= burn_in:
                history["z_samples"].append(z.copy())
                history["x_samples"].append(x.copy())

        history["last_z"] = z
        history["last_x"] = x
        history["last_log_forward"] = self._last_log_forward
        history["last_log_backward_link"] = self._last_log_backward_link
        return history
