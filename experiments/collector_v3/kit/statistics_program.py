"""Whole-block inference; no round-level independence or counterfactual regret."""

from itertools import product
import math
import numpy as np
from scipy.stats import t, chi2, nct


def paired(delta, family=1, seed=260127, resamples=20000):
    x = np.asarray(delta, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite block differences required")
    n = len(x)
    result = dict(
        blocks=n,
        mean=float(x.mean()),
        positive=int((x > 0).sum()),
        negative=int((x < 0).sum()),
        ties=int((x == 0).sum()),
    )
    if n < 2:
        return dict(
            result,
            ci_t=None,
            ci_bootstrap=None,
            ci_familywise_t=None,
            p_exchangeable=None,
            warning="One block cannot estimate between-block uncertainty",
        )
    sd = float(x.std(ddof=1))
    rng = np.random.default_rng(seed)
    samples = np.concatenate(
        [
            x[rng.integers(n, size=(min(1000, resamples - i), n))].mean(axis=1)
            for i in range(0, resamples, 1000)
        ]
    )
    means = (
        np.abs(np.array([np.mean(x * np.array(s)) for s in product((-1, 1), repeat=n)]))
        if n <= 16
        else None
    )
    if means is None:
        means = np.concatenate(
            [np.abs((x * rng.choice([-1, 1], size=(1000, n))).mean(axis=1)) for _ in range(100)]
        )
        p = float((1 + (means >= abs(x.mean()) - 1e-12).sum()) / (len(means) + 1))
    else:
        p = float((means >= abs(x.mean()) - 1e-12).mean())
    width = t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    family_width = t.ppf(1 - 0.05 / (2 * family), n - 1) * sd / math.sqrt(n)
    return dict(
        result,
        sd=sd,
        ci_t=[float(x.mean() - width), float(x.mean() + width)],
        ci_bootstrap=np.quantile(samples, [0.025, 0.975]).tolist(),
        ci_familywise_t=[float(x.mean() - family_width), float(x.mean() + family_width)],
        p_exchangeable=p,
        degenerate=sd == 0,
        assumption="Independent seed/time blocks; t intervals assume approximately normal block means; sign flips require exchangeability/symmetry under the null. Intervals conditional on these VMs.",
        warning="Zero pilot variance does not establish equivalence or zero population variance"
        if sd == 0
        else None,
    )


def holm(pvalues):
    valid = {k: v for k, v in pvalues.items() if v is not None}
    ordered = sorted(valid, key=valid.get)
    out = {k: None for k in pvalues}
    previous = 0.0
    for j, key in enumerate(ordered):
        previous = max(previous, min(1.0, (len(pvalues) - j) * valid[key]))
        out[key] = previous
    return out


def required_blocks(sd, delta, alpha=0.05, power=0.8, maximum=2000):
    if not (math.isfinite(sd) and math.isfinite(delta) and sd > 0 and delta > 0):
        return None
    for n in range(3, maximum + 1):
        cut = t.ppf(1 - alpha / 2, n - 1)
        noncentral = math.sqrt(n) * delta / sd
        actual = nct.sf(cut, n - 1, noncentral) + nct.cdf(-cut, n - 1, noncentral)
        if actual >= power:
            return n
    return None


def power_plan(delta, practical_difference, family=1):
    x = np.asarray(delta, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all():
        return dict(estimable=False, reason="At least three independent pilot blocks required")
    sd = float(x.std(ddof=1))
    if sd == 0:
        return dict(estimable=False, reason="Zero pilot variance is insufficient for planning")
    upper = sd * math.sqrt((len(x) - 1) / chi2.ppf(0.05, len(x) - 1))
    return dict(
        estimable=True,
        pilot_blocks=len(x),
        sd=sd,
        sd_upper95_normal=upper,
        practical_difference=practical_difference,
        alpha_conservative=0.05 / family,
        blocks_point_sd=required_blocks(sd, practical_difference, 0.05 / family),
        blocks_upper_sd=required_blocks(upper, practical_difference, 0.05 / family),
        caution="Prospective t-test approximation, normal independent block differences; not observed/post-hoc power and not a guarantee for a superiority margin test.",
    )
