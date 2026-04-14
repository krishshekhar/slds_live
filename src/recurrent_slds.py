"""
Recurrent switching linear dynamical system (rSLDS) with Polya-Gamma augmentation.

Linderman et al. (2017): transitions z_{t+1} | z_t, x_t use stick-breaking logits
nu = W[z_t] @ x_t + r[z_t] (weights act on the hidden continuous state, not y).

Forward-filter backward-sample for x_{1:T} fuses:
- Gaussian dynamics and observation models
- PG-induced Gaussian potentials on x_t from the augmented stick-breaking factors
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import invwishart

from .hmm_baselines import _logsumexp


def _sample_pg1(c: float | np.ndarray, trunc: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate PG(1, c) via truncated sum (Polson et al. construction)."""
    c = np.asarray(c, dtype=float)
    if c.ndim == 0:
        c = c[None]
    n = np.arange(1, trunc + 1, dtype=float)
    denom_base = (n - 0.5) ** 2
    out = np.zeros(c.shape[0], dtype=float)
    for i, ci in enumerate(c):
        gam = rng.gamma(shape=1.0, scale=1.0, size=trunc)
        denom = denom_base + (ci * ci) / (4.0 * np.pi * np.pi)
        out[i] = np.sum(gam / denom) / (2.0 * np.pi * np.pi)
    return out


def _sample_pg_b(b: int, c: float, trunc: int, rng: np.random.Generator) -> float:
    """PG(b, c) for nonnegative integer b as sum of b independent PG(1, c)."""
    if b <= 0:
        return 0.0
    return float(np.sum(_sample_pg1(np.full(b, c), trunc, rng)))


def _stick_breaking_probs(nu: np.ndarray) -> np.ndarray:
    """nu shape (..., K-1) -> probs shape (..., K)."""
    nu = np.clip(np.asarray(nu, dtype=float), -50.0, 50.0)
    sig = 1.0 / (1.0 + np.exp(-nu))
    k1 = nu.shape[-1]
    k = k1 + 1
    out = np.zeros(nu.shape[:-1] + (k,))
    rem = np.ones(nu.shape[:-1])
    for i in range(k1):
        out[..., i] = rem * sig[..., i]
        rem = rem * (1.0 - sig[..., i])
    out[..., -1] = rem
    return out


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _invert_precision_to_cov(precision: np.ndarray, *, base_ridge: float = 1e-7) -> np.ndarray:
    """
    Covariance = inv(precision) for PG-augmented linear regression.
    Uses symmetrization + ridge; falls back to pinv if still singular.
    """
    d = precision.shape[0]
    if not np.all(np.isfinite(precision)):
        return base_ridge * np.eye(d)
    p = _symmetrize(precision)
    tr = float(np.trace(p))
    ridge = max(base_ridge, 1e-9 * tr / max(d, 1))
    p_reg = p + ridge * np.eye(d)
    try:
        cov = np.linalg.inv(p_reg)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(p_reg)
    cov = _symmetrize(cov) + max(1e-10, ridge * 1e-3) * np.eye(d)
    return cov


def _log_gaussian(diff: np.ndarray, cov: np.ndarray) -> float:
    d = diff.shape[0]
    try:
        l = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        l = np.linalg.cholesky(cov + 1e-8 * np.eye(d))
    sol = np.linalg.solve(l, diff)
    quad = float(sol.T @ sol)
    logdet = 2.0 * np.sum(np.log(np.diag(l)))
    return -0.5 * (quad + logdet + d * np.log(2.0 * np.pi))


@dataclass
class RecurrentSLDS:
    """
    Finite-K rSLDS: x_t in R^M, y_t in R^D, z_t in {0..K-1}.

    x_{t+1} = A_{z_{t+1}} x_t + e_t,  e_t ~ N(0, Q_{z_{t+1}})
    y_t = C x_t + w_t,  w_t ~ N(0, R)
    z_{t+1} | z_t, x_t ~ stick_breaking( W[z_t] x_t + r[z_t] )
    """

    K: int
    state_dim: int
    obs_dim: int
    pg_trunc: int = 200
    transition_prior_var: float = 10.0
    dynamics_prior_scale: float = 1.0
    dynamics_prior_dof: float | None = None
    dynamics_prior_reg: float = 1.0
    obs_prior_scale: float = 1.0
    obs_prior_dof: float | None = None
    random_state: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        m, d = self.state_dim, self.obs_dim
        k1 = self.K - 1

        self.W = self.rng.normal(scale=0.1, size=(self.K, k1, m))
        self.r = self.rng.normal(scale=0.05, size=(self.K, k1))
        self.pi0 = np.full(self.K, 1.0 / self.K)

        self.A = self.rng.normal(scale=0.05, size=(self.K, m, m))
        self.Q = np.array([np.eye(m) for _ in range(self.K)])

        self.C = np.eye(d, m)
        self.R = np.eye(d)

        self.M0 = np.zeros((m, m))
        self.V0_inv = self.dynamics_prior_reg * np.eye(m)
        if self.dynamics_prior_dof is None:
            self.dynamics_prior_dof = m + 4.0
        self.S0 = self.dynamics_prior_scale * np.eye(m)

        if self.obs_prior_dof is None:
            self.obs_prior_dof = d + 4.0
        self.R0 = self.obs_prior_scale * np.eye(d)

    def _sample_matrix_normal(
        self, m_mat: np.ndarray, v: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        cov = np.kron(sigma, v)
        vec = self.rng.multivariate_normal(m_mat.reshape(-1, order="F"), cov)
        return vec.reshape(m_mat.shape, order="F")

    def _stick_kappa(self, z_next: int) -> np.ndarray:
        """kappa_k = I[z_next=k] - 1/2 I[z_next>=k] for k=0..K-2."""
        k1 = self.K - 1
        kappa = np.zeros(k1)
        for k in range(k1):
            kappa[k] = (1.0 if z_next == k else 0.0) - 0.5 * (
                1.0 if z_next >= k else 0.0
            )
        return kappa

    def _sample_omega(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        omega[t, k] ~ PG( I[z_{t+1}>=k], nu_{t,k} ), nu_t = W[z_t]x_t + r[z_t].
        Length T-1 along axis 0 (t = 0..T-2).
        """
        t_max = x.shape[0] - 1
        omega = np.zeros((t_max, self.K - 1))
        for t in range(t_max):
            j = z[t]
            nu = self.W[j] @ x[t] + self.r[j]
            zn = z[t + 1]
            for k in range(self.K - 1):
                b = 1 if zn >= k else 0
                omega[t, k] = _sample_pg_b(b, float(nu[k]), self.pg_trunc, self.rng)
        return omega

    def _ffbs_sample_x(
        self, y: np.ndarray, z: np.ndarray, omega: np.ndarray
    ) -> np.ndarray:
        """
        Forward filter / backward sample for x_{0:T-1} with PG Gaussian potentials on x_t
        for t = 0..T-2 fused into the Kalman update (information form) before backward pass.
        """
        t_total, m = y.shape[0], self.state_dim
        d = self.obs_dim
        i_m = np.eye(m)
        i_d = np.eye(d)

        filt_m = np.zeros((t_total, m))
        filt_p = np.zeros((t_total, m, m))
        pred_m = np.zeros((t_total, m))
        pred_p = np.zeros((t_total, m, m))

        m_prev = np.zeros(m)
        p_prev = np.eye(m)

        r_inv = np.linalg.inv(self.R + 1e-10 * i_d)
        j_obs = self.C.T @ r_inv @ self.C

        for t in range(t_total):
            if t == 0:
                m_pred = m_prev
                p_pred = p_prev
            else:
                a_t = self.A[z[t]]
                q_t = self.Q[z[t]]
                m_pred = a_t @ m_prev
                p_pred = a_t @ p_prev @ a_t.T + q_t
            pred_m[t] = m_pred
            pred_p[t] = p_pred

            try:
                p_inv = np.linalg.inv(p_pred + 1e-12 * i_m)
            except np.linalg.LinAlgError:
                p_inv = np.linalg.pinv(p_pred)

            j_f = p_inv + j_obs
            h_f = p_inv @ m_pred + self.C.T @ r_inv @ y[t]

            if t < t_total - 1:
                j_st = z[t]
                om = np.diag(omega[t])
                w_j = self.W[j_st]
                r_j = self.r[j_st]
                zn = z[t + 1]
                kappa = self._stick_kappa(zn)
                j_f = j_f + w_j.T @ om @ w_j
                h_f = h_f + w_j.T @ (kappa - om @ r_j)

            try:
                p_f = np.linalg.inv(j_f + 1e-12 * i_m)
            except np.linalg.LinAlgError:
                p_f = np.linalg.pinv(j_f)
            m_f = p_f @ h_f
            p_f = 0.5 * (p_f + p_f.T) + 1e-10 * i_m

            filt_m[t] = m_f
            filt_p[t] = p_f
            m_prev, p_prev = m_f, p_f

        x = np.zeros((t_total, m))
        
        cov_last = 0.5 * (filt_p[-1] + filt_p[-1].T)
        jitter = 1e-5
        while True:
            try:
                np.linalg.cholesky(cov_last + jitter * i_m)
                cov_last = cov_last + jitter * i_m
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
                
        x[-1] = self.rng.multivariate_normal(filt_m[-1], cov_last)
        for t in range(t_total - 2, -1, -1):
            a_next = self.A[z[t + 1]]
            p_pn = pred_p[t + 1]
            try:
                p_pn_inv = np.linalg.inv(p_pn + 1e-12 * i_m)
            except np.linalg.LinAlgError:
                p_pn_inv = np.linalg.pinv(p_pn)
            j_gain = filt_p[t] @ a_next.T @ p_pn_inv
            mean = filt_m[t] + j_gain @ (x[t + 1] - pred_m[t + 1])
            cov = filt_p[t] - j_gain @ p_pn @ j_gain.T
            cov = 0.5 * (cov + cov.T)
            
            jitter = 1e-5
            while True:
                try:
                    np.linalg.cholesky(cov + jitter * i_m)
                    cov = cov + jitter * i_m
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
                    
            x[t] = self.rng.multivariate_normal(mean, cov)
        return x

    def _transition_log_probs(self, x: np.ndarray) -> np.ndarray:
        """log P(z_{t+1}=k | z_t=j, x_t), shape (T-1, K, K)."""
        t_max = x.shape[0] - 1
        out = np.zeros((t_max, self.K, self.K))
        for t in range(t_max):
            for j in range(self.K):
                nu = self.W[j] @ x[t] + self.r[j]
                probs = _stick_breaking_probs(nu)
                out[t, j, :] = np.log(probs + 1e-32)
        return out

    def _sample_z(self, x: np.ndarray) -> np.ndarray:
        """Block sample z | x using forward-filter backward-sample (Markov on z given x)."""
        t_total = x.shape[0]
        log_trans = self._transition_log_probs(x)

        log_alpha = np.zeros((t_total, self.K))
        log_alpha[0] = np.log(self.pi0 + 1e-32)
        for t in range(t_total - 1):
            tmp = log_alpha[t][:, None] + log_trans[t]
            log_alpha[t + 1] = _logsumexp(tmp, axis=0)

        z = np.zeros(t_total, dtype=int)
        p_last = np.exp(log_alpha[-1] - _logsumexp(log_alpha[-1]))
        z[-1] = self.rng.choice(self.K, p=p_last)
        for t in range(t_total - 2, -1, -1):
            lp = log_alpha[t] + log_trans[t, :, z[t + 1]]
            p_t = np.exp(lp - _logsumexp(lp))
            z[t] = self.rng.choice(self.K, p=p_t)
        return z

    def _sample_dynamics(self, x: np.ndarray, z: np.ndarray) -> None:
        for k in range(self.K):
            idx = np.where(z[1:] == k)[0] + 1
            if idx.size == 0:
                self.Q[k] = invwishart.rvs(df=self.dynamics_prior_dof, scale=self.S0)
                self.A[k] = self._sample_matrix_normal(
                    self.M0, np.linalg.inv(self.V0_inv), self.Q[k]
                )
                continue
            xk_prev = x[idx - 1]
            xk = x[idx]
            vn_inv = self.V0_inv + xk_prev.T @ xk_prev
            vn_inv = 0.5 * (vn_inv + vn_inv.T) # Force symmetry
            
            # Try strict inverse, fallback to safe pseudo-inverse
            try:
                vn = np.linalg.inv(vn_inv + 1e-5 * np.eye(self.state_dim))
            except np.linalg.LinAlgError:
                vn = np.linalg.pinv(vn_inv + 1e-5 * np.eye(self.state_dim))
                
            vn = 0.5 * (vn + vn.T) # Ensure the resulting covariance is also symmetric
            mn = vn @ (self.V0_inv @ self.M0 + xk_prev.T @ xk)
            sn = (
                self.S0
                + xk.T @ xk
                + self.M0.T @ self.V0_inv @ self.M0
                - mn.T @ vn_inv @ mn
            )
            sn = 0.5 * (sn + sn.T)
            
            # Dynamic Jitter: Exponentially increase noise until matrix is strictly positive-definite
            jitter = 1e-5
            while True:
                try:
                    np.linalg.cholesky(sn + jitter * np.eye(self.state_dim))
                    sn = sn + jitter * np.eye(self.state_dim)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0

            nun = self.dynamics_prior_dof + idx.size
            self.Q[k] = invwishart.rvs(df=nun, scale=sn)
            bk = self._sample_matrix_normal(mn, vn, self.Q[k])
            self.A[k] = bk.T

    def _sample_observation_noise(self, y: np.ndarray, x: np.ndarray) -> None:
        resid = y - (self.C @ x.T).T
        sr = resid.T @ resid + self.R0
        
        # 1. Force perfect symmetry
        sr = 0.5 * (sr + sr.T)
        
        # 2. Dynamic Jitter: Exponentially increase noise until strictly positive-definite
        jitter = 1e-5
        while True:
            try:
                np.linalg.cholesky(sr + jitter * np.eye(self.obs_dim))
                sr = sr + jitter * np.eye(self.obs_dim)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
                
        dof = self.obs_prior_dof + y.shape[0]
        self.R = invwishart.rvs(df=dof, scale=sr)

    def _sample_recurrence(self, x: np.ndarray, z: np.ndarray) -> None:
        """
        Bayesian linear regression for each row of stick-breaking, with design [x_t, 1].
        PG augmentation matches Linderman et al.; features come from latent x_t only.
        """
        t_max = x.shape[0] - 1
        feat_dim = self.state_dim + 1
        prior_prec = 1.0 / self.transition_prior_var

        for j in range(self.K):
            for k in range(self.K - 1):
                idx_list: list[int] = []
                for t in range(t_max):
                    if z[t] != j or z[t + 1] < k:
                        continue
                    idx_list.append(t)
                if not idx_list:
                    continue
                ph = np.stack([np.concatenate([x[t], [1.0]]) for t in idx_list], axis=0)
                yb = np.array([1.0 if z[t + 1] == k else 0.0 for t in idx_list])
                w_old = np.concatenate([self.W[j, k], [self.r[j, k]]])
                c_vec = ph @ w_old
                omega = np.zeros(len(idx_list))
                for i, t in enumerate(idx_list):
                    b_pg = 1 if z[t + 1] >= k else 0
                    omega[i] = _sample_pg_b(b_pg, float(c_vec[i]), self.pg_trunc, self.rng)
                kappa = yb - 0.5
                omega = np.maximum(omega, 1e-12)
                xt_omega = ph.T * omega[None, :]
                precision = xt_omega @ ph + prior_prec * np.eye(feat_dim)
                precision = 0.5 * (precision + precision.T) # Force perfect symmetry
                
                # Try strict inverse, fallback to safe pseudo-inverse
                try:
                    cov_w = np.linalg.inv(precision + 1e-5 * np.eye(feat_dim))
                except np.linalg.LinAlgError:
                    cov_w = np.linalg.pinv(precision + 1e-5 * np.eye(feat_dim))
                    
                cov_w = 0.5 * (cov_w + cov_w.T) + 1e-6 * np.eye(feat_dim) # Safety jitter
                mean_w = cov_w @ (ph.T @ kappa)
                if not np.all(np.isfinite(mean_w)):
                    continue
                w_new = self.rng.multivariate_normal(mean_w, cov_w)
                self.W[j, k] = w_new[:-1]
                self.r[j, k] = w_new[-1]

    def _complete_data_log_score(
        self, y: np.ndarray, x: np.ndarray, z: np.ndarray
    ) -> float:
        ll = 0.0
        for t in range(y.shape[0]):
            ll += _log_gaussian(y[t] - self.C @ x[t], self.R)
        for t in range(1, y.shape[0]):
            mean = self.A[z[t]] @ x[t - 1]
            ll += _log_gaussian(x[t] - mean, self.Q[z[t]])
        for t in range(y.shape[0] - 1):
            j = z[t]
            nu = self.W[j] @ x[t] + self.r[j]
            probs = _stick_breaking_probs(nu)
            ll += float(np.log(probs[z[t + 1]] + 1e-32))
        return ll

    def gibbs(
        self,
        y: ArrayLike,
        n_iters: int = 200,
        burn_in: int = 100,
        *,
        x_init: np.ndarray | None = None,
        z_init: np.ndarray | None = None,
    ) -> dict:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[1] != self.obs_dim:
            raise ValueError("obs_dim does not match y.shape[1].")

        t_total = y.shape[0]
        if x_init is not None or z_init is not None:
            if x_init is None or z_init is None:
                raise ValueError("Provide both x_init and z_init, or neither.")
            x = np.asarray(x_init, dtype=float).copy()
            z = np.asarray(z_init, dtype=int).copy()
            if x.shape != (t_total, self.state_dim):
                raise ValueError(
                    f"x_init must have shape ({t_total}, {self.state_dim}), got {x.shape}."
                )
            if z.shape != (t_total,):
                raise ValueError(f"z_init must have shape ({t_total},), got {z.shape}.")
        else:
            z = self.rng.integers(0, self.K, size=t_total)
            x = self.rng.normal(size=(t_total, self.state_dim))
        omega = np.zeros((max(0, t_total - 1), self.K - 1))

        history: dict[str, list] = {"loglik": [], "z_samples": [], "x_samples": []}

        for it in range(n_iters):
            omega = self._sample_omega(x, z)
            x = self._ffbs_sample_x(y, z, omega)
            z = self._sample_z(x)
            self._sample_dynamics(x, z)
            self._sample_observation_noise(y, x)
            self._sample_recurrence(x, z)

            history["loglik"].append(self._complete_data_log_score(y, x, z))
            if it >= burn_in:
                history["z_samples"].append(z.copy())
                history["x_samples"].append(x.copy())

        history["last_z"] = z
        history["last_x"] = x
        history["last_omega"] = omega
        return history
