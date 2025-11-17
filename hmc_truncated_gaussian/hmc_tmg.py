"""Modernized HMC sampler for truncated multivariate Gaussians."""

from __future__ import annotations

import sys
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

EPS = 1e-11


def _as_column(vec: Sequence[float]) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=float)
    return arr.reshape(arr.size, 1)


class HMCTruncGaussian:
    """Hamiltonian Monte Carlo sampler for truncated multivariate Gaussians."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def _draw_velocity(self, dim: int) -> NDArray[np.float64]:
        return self.rng.normal(size=(dim, 1))

    def generate_simple_tmg(
        self,
        mean: Sequence[float],
        std_dev: float,
        samples: int = 1,
    ) -> list[list[float]]:
        dim = len(mean)
        mu = _as_column(mean)
        if (mu < 0).any():
            raise ValueError("mean vector must be positive")
        if std_dev <= 0:
            raise ValueError("standard deviation must be positive")

        g = mu.copy()
        initial_sample = (np.ones((dim, 1)) - mu) / std_dev
        if (initial_sample + g < 0).any():
            raise ValueError("inconsistent initial condition")

        sample_matrix: list[list[float]] = []
        for _ in range(samples):
            stop = False
            j = -1
            initial_velocity = self._draw_velocity(dim)
            x = initial_sample.copy()
            T = np.pi / 2
            tt = 0.0

            while True:
                a = np.real(initial_velocity.copy())
                b = x.copy()

                u = np.sqrt((std_dev**2) * (a**2 + b**2))
                phi = np.arctan2(-(std_dev**2) * a, (std_dev**2) * b)

                pn = np.abs(g / u)
                t1 = np.full((dim, 1), np.inf)

                collision = False
                inds = [-1] * dim
                for k in range(dim):
                    if pn[k] <= 1:
                        collision = True
                        pn[k] = 1
                        t1[k] = -phi[k] + np.arccos((-g[k]) / u[k])
                        inds[k] = k
                    else:
                        pn[k] = 0

                if collision:
                    if j > -1 and pn[j] == 1:
                        cum_sum_pn = np.cumsum(pn, axis=0)
                        index_j = int(cum_sum_pn[j, 0] - 1)
                        tt1 = t1[index_j]
                        if np.abs(tt1) < EPS or np.abs(tt1 - 2 * np.pi) < EPS:
                            t1[index_j] = np.inf

                    mt = np.min(t1)
                    j = inds[int(np.argmin(t1))]
                else:
                    mt = T

                tt += mt
                if tt >= T:
                    mt -= tt - T
                    stop = True

                x = a * np.sin(mt) + b * np.cos(mt)
                v = a * np.cos(mt) - b * np.sin(mt)

                if stop:
                    break

                initial_velocity[j] = -v[j]
                for k in range(dim):
                    if k != j:
                        initial_velocity[k] = v[k]

            sample = std_dev * x + mu
            sample_matrix.append(sample.ravel().tolist())

        return sample_matrix

    def generate_general_tmg(
        self,
        fc: NDArray[np.float64],
        gc: NDArray[np.float64],
        m: NDArray[np.float64],
        mean_r: Sequence[float],
        initial: NDArray[np.float64],
        samples: int = 1,
        cov: bool = True,
    ) -> list[list[float]]:
        fc = np.asarray(fc, dtype=float)
        gc = _as_column(gc)
        initial = np.asarray(initial, dtype=float)

        if fc.shape[0] != gc.shape[0]:
            raise ValueError("constraint dimensions do not match")

        try:
            r = np.linalg.cholesky(m)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance or precision matrix is not positive definite") from exc

        if cov:
            mu = _as_column(mean_r)
            g = gc + fc @ mu
            f = fc @ r.T
            initial_sample = np.linalg.solve(r.T, initial - mu)
        else:
            r_vec = _as_column(mean_r)
            mu = np.linalg.solve(r, np.linalg.solve(r.T, r_vec))
            g = gc + fc @ mu
            f = np.linalg.solve(r, fc)
            initial_sample = r @ (initial - mu)

        if (f @ initial_sample + g < 0).any():
            raise ValueError("inconsistent initial condition")

        dim = mu.shape[0]
        sample_matrix: list[list[float]] = []
        fsq = np.sum(f**2, axis=0, keepdims=True)
        ft = f.T

        for _ in range(samples):
            stop = False
            j = -1
            initial_velocity = self._draw_velocity(dim)
            x = initial_sample.copy()
            T = np.pi / 2
            tt = 0.0

            while True:
                a = np.real(initial_velocity.copy())
                b = x.copy()

                fa = f @ a
                fb = f @ b
                u = np.sqrt(fa**2 + fb**2)
                phi = np.arctan2(-fa, fb)

                pn = np.abs(g / u)
                t1 = np.full_like(pn, np.inf)

                collision = False
                inds = [-1] * pn.shape[0]
                for k in range(pn.shape[0]):
                    if pn[k] <= 1:
                        collision = True
                        pn[k] = 1
                        t1[k] = -phi[k] + np.arccos((-g[k]) / u[k])
                        inds[k] = k
                    else:
                        pn[k] = 0

                if collision:
                    if j > -1 and pn[j] == 1:
                        cum_sum_pn = np.cumsum(pn, axis=0)
                        index_j = int(cum_sum_pn[j, 0] - 1)
                        tt1 = t1[index_j]
                        if np.abs(tt1) < EPS or np.abs(tt1 - 2 * np.pi) < EPS:
                            t1[index_j] = np.inf

                    mt = float(np.min(t1))
                    j = inds[int(np.argmin(t1))]
                else:
                    mt = T

                tt += mt
                if tt >= T:
                    mt -= tt - T
                    stop = True

                x = a * np.sin(mt) + b * np.cos(mt)
                v = a * np.cos(mt) - b * np.sin(mt)

                if stop:
                    break

                reflected = (f[j, :] @ v) / fsq[0, j]
                initial_velocity = v - 2 * reflected * ft[:, [j]]

            if cov:
                sample = r.T @ x + mu
            else:
                sample = np.linalg.solve(r, x) + mu

            sample_matrix.append(sample.ravel().tolist())

        return sample_matrix
