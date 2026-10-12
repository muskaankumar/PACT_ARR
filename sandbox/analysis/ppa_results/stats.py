"""
Dependency-free-ish statistical primitives for the PPA analysis.

Everything here is implemented explicitly rather than pulled from a stats
package so that every number in the paper can be traced to a formula.
Only numpy is required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------- normal dist


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's algorithm, |err| < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _norm_sf(z: float) -> float:
    """Upper-tail standard-normal survival function."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_sided_p(z: float) -> float:
    return 2.0 * _norm_sf(abs(z))


# ------------------------------------------------------------ binomial CIs


@dataclass
class Proportion:
    k: int
    n: int
    p: float
    lo: float
    hi: float

    def pct(self) -> str:
        return f"{100*self.p:.1f}% [{100*self.lo:.1f}, {100*self.hi:.1f}]"

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def wilson(k: int, n: int, alpha: float = 0.05) -> Proportion:
    """Wilson score interval for a binomial proportion."""
    k, n = int(k), int(n)
    if n == 0:
        return Proportion(0, 0, float("nan"), float("nan"), float("nan"))
    z = _norm_ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Proportion(k, n, p, max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class TwoProportionTest:
    k1: int; n1: int; p1: float
    k2: int; n2: int; p2: float
    diff_pp: float
    diff_lo_pp: float
    diff_hi_pp: float
    ratio: float
    z: float
    p_value: float
    cohens_h: float

    def as_dict(self) -> dict:
        return asdict(self)


def two_proportion_test(k1: int, n1: int, k2: int, n2: int,
                        alpha: float = 0.05) -> TwoProportionTest:
    """
    Group 1 minus group 2. Pooled-variance z-test for the null of equal
    proportions; unpooled Wald interval for the difference (standard practice:
    test pooled, interval unpooled).
    """
    p1 = k1 / n1 if n1 else float("nan")
    p2 = k2 / n2 if n2 else float("nan")
    diff = p1 - p2
    z_crit = _norm_ppf(1 - alpha / 2)

    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 and n2 else float("nan")
    lo, hi = diff - z_crit * se_unpooled, diff + z_crit * se_unpooled

    p_pool = (k1 + k2) / (n1 + n2)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = diff / se_pool if se_pool > 0 else float("nan")
    pval = two_sided_p(z) if se_pool > 0 else float("nan")

    ratio = (p1 / p2) if p2 > 0 else float("inf")
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

    return TwoProportionTest(k1, n1, p1, k2, n2, p2,
                             100 * diff, 100 * lo, 100 * hi,
                             ratio, z, pval, h)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """
    Two-sided Fisher exact p-value for [[a,b],[c,d]] by summing all tables
    with probability <= the observed table's probability. Exact, no scipy.
    """
    from math import comb
    r1, r2 = a + b, c + d
    c1 = a + c
    n = r1 + r2
    total = comb(n, c1)

    def prob(x):
        if x < 0 or x > r1 or (c1 - x) < 0 or (c1 - x) > r2:
            return 0.0
        return comb(r1, x) * comb(r2, c1 - x) / total

    p_obs = prob(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    tol = 1e-12
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= p_obs + tol))


# --------------------------------------------------------------- agreement


@dataclass
class KappaResult:
    n: int
    categories: list
    observed_agreement: float
    expected_agreement: float
    kappa: float
    se: float
    lo: float
    hi: float
    weighting: str
    interpretation: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["categories"] = "|".join(map(str, d["categories"]))
        return d


def _landis_koch(k: float) -> str:
    if k != k:
        return "undefined"
    if k < 0:      return "poor (worse than chance)"
    if k < 0.21:   return "slight"
    if k < 0.41:   return "fair"
    if k < 0.61:   return "moderate"
    if k < 0.81:   return "substantial"
    return "almost perfect"


def confusion_matrix(a: Sequence, b: Sequence, categories: Sequence) -> np.ndarray:
    idx = {c: i for i, c in enumerate(categories)}
    m = np.zeros((len(categories), len(categories)), dtype=float)
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    return m


def cohens_kappa(a: Sequence, b: Sequence,
                 categories: Sequence | None = None,
                 weights: str = "unweighted",
                 alpha: float = 0.05) -> KappaResult:
    """
    Cohen's kappa with an asymptotic standard error.

    weights: 'unweighted' | 'linear' | 'quadratic'
      Weighted variants require `categories` to be given in ordinal order.
      SE for weighted kappa uses the standard Fleiss-Cohen-Everitt
      approximation; for the unweighted case it reduces to Fleiss's formula.
    """
    a, b = list(a), list(b)
    if categories is None:
        categories = sorted(set(a) | set(b))
    categories = list(categories)
    n = len(a)
    if n == 0:
        return KappaResult(0, categories, float("nan"), float("nan"),
                           float("nan"), float("nan"), float("nan"),
                           float("nan"), weights, "undefined")

    k = len(categories)
    O = confusion_matrix(a, b, categories) / n
    row = O.sum(axis=1)
    col = O.sum(axis=0)
    E = np.outer(row, col)

    # AGREEMENT weights (diagonal = 1), the standard formulation
    if weights == "unweighted":
        W = np.eye(k)
    elif weights == "linear":
        W = 1.0 - np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    elif weights == "quadratic":
        W = 1.0 - (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / ((k - 1) ** 2)
    else:
        raise ValueError(weights)

    po = float((W * O).sum())
    pe = float((W * E).sum())
    obs_agree, exp_agree = po, pe
    kappa = float("nan") if pe == 1 else (po - pe) / (1 - pe)

    # Asymptotic SE (Fleiss, Cohen & Everitt 1969)
    if pe == 1 or n <= 1 or kappa != kappa:
        se = float("nan")
    else:
        wbar_i = (W * col[None, :]).sum(axis=1)   # row i marginal weight
        wbar_j = (W * row[:, None]).sum(axis=0)   # col j marginal weight
        term = 0.0
        for i in range(k):
            for j in range(k):
                term += O[i, j] * (W[i, j]
                                   - (wbar_i[i] + wbar_j[j]) * (1 - kappa)) ** 2
        var = (term - (kappa - pe * (1 - kappa)) ** 2) / (n * (1 - pe) ** 2)
        se = math.sqrt(var) if var > 0 else 0.0

    z = _norm_ppf(1 - alpha / 2)
    lo = kappa - z * se if se == se else float("nan")
    hi = kappa + z * se if se == se else float("nan")

    return KappaResult(n, categories, obs_agree, exp_agree, kappa, se,
                       lo, hi, weights, _landis_koch(kappa))


# ------------------------------------------------------- Kaplan-Meier


@dataclass
class KMCurve:
    times: np.ndarray
    survival: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n_risk: np.ndarray
    n_event: np.ndarray
    median: float
    n: int
    n_events: int


def kaplan_meier(durations: Sequence[float], events: Sequence[bool],
                 alpha: float = 0.05) -> KMCurve:
    """
    Kaplan-Meier survivor function with Greenwood variance and
    log-log transformed pointwise confidence limits.
    """
    d = np.asarray(list(durations), dtype=float)
    e = np.asarray(list(events), dtype=bool)
    order = np.argsort(d, kind="mergesort")
    d, e = d[order], e[order]
    n = len(d)

    uniq = np.unique(d[e]) if e.any() else np.array([])
    times, surv, low, up, at_risk, n_ev = [0.0], [1.0], [1.0], [1.0], [n], [0]
    s = 1.0
    cum_var = 0.0
    for t in uniq:
        n_i = int((d >= t).sum())
        d_i = int(((d == t) & e).sum())
        if n_i == 0:
            continue
        s *= (1 - d_i / n_i)
        if n_i - d_i > 0:
            cum_var += d_i / (n_i * (n_i - d_i))
        times.append(float(t)); surv.append(s); at_risk.append(n_i); n_ev.append(d_i)
        if 0 < s < 1 and cum_var > 0:
            z = _norm_ppf(1 - alpha / 2)
            se_loglog = math.sqrt(cum_var) / abs(math.log(s))
            lo = s ** math.exp(z * se_loglog)
            hi = s ** math.exp(-z * se_loglog)
        else:
            lo, hi = s, s
        low.append(lo); up.append(hi)

    surv_arr = np.array(surv)
    times_arr = np.array(times)
    med = float("nan")
    below = np.where(surv_arr <= 0.5)[0]
    if below.size:
        med = float(times_arr[below[0]])

    return KMCurve(times_arr, surv_arr, np.array(low), np.array(up),
                   np.array(at_risk), np.array(n_ev), med, n, int(e.sum()))


def logrank_test(d1, e1, d2, e2) -> tuple:
    """Two-sample log-rank test. Returns (chi2, p, observed1, expected1)."""
    d1 = np.asarray(list(d1), float); e1 = np.asarray(list(e1), bool)
    d2 = np.asarray(list(d2), float); e2 = np.asarray(list(e2), bool)
    all_times = np.unique(np.concatenate([d1[e1], d2[e2]]))
    O1 = E1 = V = 0.0
    for t in all_times:
        n1 = (d1 >= t).sum(); n2 = (d2 >= t).sum()
        o1 = ((d1 == t) & e1).sum(); o2 = ((d2 == t) & e2).sum()
        n, o = n1 + n2, o1 + o2
        if n < 2 or o == 0:
            continue
        E1 += o * n1 / n
        O1 += o1
        V += o * (n1 / n) * (1 - n1 / n) * ((n - o) / (n - 1))
    if V <= 0:
        return float("nan"), float("nan"), O1, E1
    chi2 = (O1 - E1) ** 2 / V
    p = _norm_sf(math.sqrt(chi2)) * 2
    return chi2, p, O1, E1


# ------------------------------------------------------------- misc helpers


def bootstrap_ci(values: Sequence[float], stat=np.mean, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 20260703) -> tuple:
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = stat(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    return float(stat(v)), float(np.quantile(draws, alpha / 2)), \
           float(np.quantile(draws, 1 - alpha / 2))


def mann_whitney_u(x, y) -> tuple:
    """Normal-approximation Mann-Whitney U with tie correction."""
    x = np.asarray(list(x), float); y = np.asarray(list(y), float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    combined = np.concatenate([x, y])
    order = combined.argsort()
    ranks = np.empty(len(combined), float)
    ranks[order] = np.arange(1, len(combined) + 1)
    # average ties
    srt = combined[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    _, counts = np.unique(combined, return_counts=True)
    tie_term = (counts ** 3 - counts).sum()
    N = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((N + 1) - tie_term / (N * (N - 1))))
    if sd == 0:
        return U1, float("nan"), float("nan")
    z = (U1 - mu) / sd
    return U1, z, two_sided_p(z)


def chi2_independence(table: np.ndarray) -> tuple:
    """Pearson chi-square test of independence. Returns (chi2, df, p, cramers_v)."""
    obs = np.asarray(table, float)
    obs = obs[obs.sum(axis=1) > 0][:, obs.sum(axis=0) > 0]
    n = obs.sum()
    if n == 0 or obs.shape[0] < 2 or obs.shape[1] < 2:
        return float("nan"), 0, float("nan"), float("nan")
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    df = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    p = _chi2_sf(chi2, df)
    v = math.sqrt(chi2 / (n * (min(obs.shape) - 1)))
    return chi2, df, p, v


def _chi2_sf(x: float, df: int) -> float:
    """Upper tail of chi-square via regularised incomplete gamma (series/CF)."""
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1:
        # series for P(a,x)
        term = 1.0 / a
        s = term
        n_ = 1
        while n_ < 1000:
            term *= xx / (a + n_)
            s += term
            if abs(term) < abs(s) * 1e-14:
                break
            n_ += 1
        return 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    # continued fraction for Q(a,x)
    tiny = 1e-300
    b = xx + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny: d = tiny
        c = b + an / c
        if abs(c) < tiny: c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h
