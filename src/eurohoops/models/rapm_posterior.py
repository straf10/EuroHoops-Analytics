"""M3 H4: closed-form Bayesian posterior for RAPM ratings (PLAN §5.4, H-g).

Subagent A's design (``models/rapm.py``) fits O/D-RAPM per H-b by weighted ridge: one row per
stint side, target points per 100 possessions, weight possessions (times the time decay), a
ridge penalty per column toward a prior mean (0 for most columns, ~0 for the unpenalised
intercept and home column). H-g asks for the Bayesian read of that same fit: a Normal-Normal
conjugate model where the ridge solution is the posterior mean and a 90% interval comes from the
posterior covariance, closed form (no PyMC, an explicit deviation from PLAN wording the owner
chose; see ``reports/week9-12_progress.md`` §0).

Model, one column j per rating (and the intercept/home columns with an ~0 penalty):

    y_i ~ N(x_i . theta, sigma^2 / w_i)                     (w_i = possessions * time decay)
    theta_j ~ N(prior_mean_j, sigma^2 / penalty_j)           (penalty_j = sigma^2 / tau_j^2)

Conjugacy gives, in the normal equations ``gram = X^T W X`` and ``rhs = X^T W y``:

    posterior mean       = (gram + diag(penalty))^-1 (rhs + diag(penalty) . prior_mean)
    posterior covariance = sigma^2 (gram + diag(penalty))^-1

The posterior mean is exactly the weighted ridge solution toward ``prior_mean``; a column with
``penalty[j] ~ 0`` (e.g. 1e-8, the intercept/home convention ``models/team_eff.py`` also uses) is
effectively flat (an improper uniform prior), and its posterior sd converges to the OLS standard
error of that column.

``posterior_from_normal_equations`` is the function subagent A's walk-forward fit calls: A keeps
the decayed normal equations already (mirroring ``DecayedRidge`` in ``models/team_eff.py``), so
no design matrix needs to be rebuilt. ``posterior`` is the same computation from a raw sparse
design, for callers (and tests) that have not formed the normal equations yet. ``noise_variance``
gives the orchestrator sigma^2 from residuals of the tuning fit. ``ridge_solution`` is an
independent reference solve (augmented least squares, not the Cholesky-on-normal-equations path
below), used only to check that path in tests.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import linalg, sparse, special

from eurohoops.models.elo import FloatArray

# A dense p x p Cholesky (and its inverse) is the whole point at RAPM's scale: ~3,456 columns
# (1,727 players x O/D) is ~95 MB and a few seconds (reports/week9-12_progress.md §3.1). Above
# this, forming and factoring a dense p x p matrix stops being reasonable; a sparse/iterative
# solve (CG) would be needed instead, which this module does not implement (H-g: "sparse/CG if
# not" — not required at M3's player count).
MAX_DENSE_PARAMS = 6_000


@dataclass(frozen=True)
class Posterior:
    """Posterior mean and marginal standard deviation of each rating, from a Normal-Normal fit.

    The joint posterior is multivariate Normal with a full covariance (ratings are correlated
    through shared lineups); ``sd`` keeps only the marginal (diagonal) standard deviations, all
    H-g's per-player 90% intervals need.
    """

    mean: FloatArray
    sd: FloatArray

    def __post_init__(self) -> None:
        if self.mean.shape != self.sd.shape:
            raise ValueError(f"mean and sd shapes differ: {self.mean.shape} vs {self.sd.shape}")

    def interval(self, level: float = 0.9) -> tuple[FloatArray, FloatArray]:
        """Symmetric Normal interval, e.g. ``level=0.9`` -> the middle 90% around ``mean``."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        z = float(special.ndtri(0.5 + level / 2.0))
        margin = z * self.sd
        return self.mean - margin, self.mean + margin


def _as_dense(matrix: sparse.spmatrix | FloatArray) -> FloatArray:
    if isinstance(matrix, sparse.spmatrix):
        # `spmatrix` itself is an untyped mixin (`.toarray` lives on the concrete subclasses,
        # e.g. `csr_matrix`, all of which have it); any real instance does have it at runtime.
        dense: FloatArray = cast("sparse.csr_matrix", matrix).toarray()
        return dense.astype(np.float64, copy=False)
    return np.asarray(matrix, dtype=np.float64)


def posterior_from_normal_equations(
    gram: sparse.spmatrix | FloatArray,
    rhs: FloatArray,
    penalty: FloatArray,
    prior_mean: FloatArray,
    noise_var: float,
) -> Posterior:
    """Posterior from the normal equations ``gram = X^T W X``, ``rhs = X^T W y`` (p x p, p).

    Dense Cholesky (p <= ``MAX_DENSE_PARAMS``); raises ``ValueError`` above that. The posterior
    covariance's diagonal is read off the explicit inverse (``cho_solve`` against the identity):
    at M3's scale (~3,500 columns) that inverse is the ~95 MB the module docstring budgets for,
    and computing it through the Cholesky factor (rather than ``numpy.linalg.inv``) reuses the
    one factorisation ``mean`` also needs and stays numerically stable for a near-singular
    system (a column seen in very few stints, close to its unpenalised prior).
    """
    p = rhs.shape[0]
    if penalty.shape != (p,) or prior_mean.shape != (p,):
        raise ValueError(
            f"shape mismatch: rhs {rhs.shape}, penalty {penalty.shape}, "
            f"prior_mean {prior_mean.shape}"
        )
    if gram.shape != (p, p):
        raise ValueError(f"gram shape {gram.shape} does not match rhs length {p}")
    if p > MAX_DENSE_PARAMS:
        raise ValueError(
            f"posterior_from_normal_equations: p={p} exceeds the dense limit "
            f"({MAX_DENSE_PARAMS}); a sparse/CG solve is needed at this scale"
        )
    system = _as_dense(gram) + np.diag(penalty)
    adjusted_rhs = rhs + penalty * prior_mean
    factor = linalg.cho_factor(system, lower=True)
    mean: FloatArray = linalg.cho_solve(factor, adjusted_rhs)
    inverse: FloatArray = linalg.cho_solve(factor, np.eye(p))
    variance = noise_var * np.diag(inverse)
    sd = np.sqrt(np.clip(variance, 0.0, None))
    return Posterior(mean=mean, sd=sd)


def posterior(
    x: sparse.csr_matrix,
    weight: FloatArray,
    target: FloatArray,
    *,
    penalty: FloatArray,
    prior_mean: FloatArray,
    noise_var: float,
) -> Posterior:
    """Posterior from the design directly: ``x`` (n x p, sparse CSR), row weight, target ``y``.

    Forms the normal equations (``gram = X^T W X``, ``rhs = X^T W y``) and calls
    ``posterior_from_normal_equations``. Subagent A's walk-forward fit keeps decayed normal
    equations already (like ``DecayedRidge`` in ``models/team_eff.py``) and should call that
    function directly instead of rebuilding ``x`` each round; this entry point is for a one-shot
    fit from a raw design (tests, and any caller without incremental state).
    """
    if x.shape[0] != weight.shape[0] or x.shape[0] != target.shape[0]:
        raise ValueError(
            f"row count mismatch: x {x.shape[0]}, weight {weight.shape[0]}, "
            f"target {target.shape[0]}"
        )
    weighted = x.multiply(weight[:, None]).tocsr()
    gram = (weighted.T @ x).toarray()
    rhs: FloatArray = np.asarray(weighted.T @ target, dtype=np.float64).ravel()
    return posterior_from_normal_equations(gram, rhs, penalty, prior_mean, noise_var)


def ridge_solution(
    x: sparse.csr_matrix | FloatArray,
    weight: FloatArray,
    target: FloatArray,
    penalty: FloatArray,
    prior_mean: FloatArray,
) -> FloatArray:
    """The same weighted ridge solution, solved independently for tests (augmented least
    squares via QR, not the Cholesky-on-normal-equations path ``posterior`` uses), so comparing
    ``posterior(...).mean`` against this is a check of that path rather than the same code twice:

        minimise || sqrt(w) . (y - X theta) ||^2 + || sqrt(penalty) . (theta - prior_mean) ||^2

    is a plain least squares problem in the stacked design ``[sqrt(w) X; diag(sqrt(penalty))]``.
    """
    p = prior_mean.shape[0]
    sqrt_w = np.sqrt(weight)
    sqrt_penalty = np.sqrt(penalty)
    x_dense = _as_dense(x)
    design = np.vstack([x_dense * sqrt_w[:, None], np.diag(sqrt_penalty)])
    stacked_target = np.concatenate([sqrt_w * target, sqrt_penalty * prior_mean])
    solution, _residuals, _rank, _singular = np.linalg.lstsq(design, stacked_target, rcond=None)
    result: FloatArray = solution.reshape(p)
    return result


def noise_variance(residual: FloatArray, weight: FloatArray, dof: float) -> float:
    """sigma^2 = sum(w_i * r_i^2) / (n - dof), from the tuning fit's residuals.

    ``dof`` (degrees of freedom used) is the caller's choice; we recommend passing the number of
    fitted rating columns ``p`` rather than the ridge's effective dof ``trace(H)``
    (``H = X (X^T W X + diag(penalty))^-1 X^T W``): at M3's tuning scale n (stint-side rows) is
    orders of magnitude bigger than p (~3,500 columns), so the two are close, and ``p`` avoids
    forming ``H`` (an n x p x p computation the orchestrator would otherwise pay for just to get
    one scalar). Because ridge shrinkage only ever pulls ``trace(H)`` below ``p``, using ``p``
    slightly over-counts dof and so slightly *overestimates* sigma^2 — conservative (wider
    intervals) rather than the undercoverage a too-small sigma^2 would cause.
    """
    n = residual.shape[0]
    denom = n - dof
    if denom <= 0:
        raise ValueError(f"dof={dof} leaves no degrees of freedom for n={n} residuals")
    return float(np.sum(weight * residual**2) / denom)
