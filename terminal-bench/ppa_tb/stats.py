"""Statistics for the paper tables. Stdlib only.

One thing here is not optional for publication. In the v3 sandbox every episode
ran on one repo, so episodes were exchangeable and Wilson intervals on the raw
counts were honest. In the Terminal-Bench wrapper, 120 episodes sit on 10 tasks,
12 apiece, and episodes within a task are strongly correlated: a task whose
overlay is salient produces few violations across every model and arm, a task
whose forbidden path sits in the working directory produces many.

Treating those 120 as independent understates the standard error by roughly
sqrt(1 + (m-1) * rho). At m = 12 and a plausible rho of 0.2, that is a factor of
~1.7 — enough to turn a null into a "finding". So every headline interval in the
paper comes from a **cluster bootstrap that resamples tasks, not episodes**, and
the naive interval is reported alongside only to show the gap.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Naive (episode-level) interval. Correct only if episodes are exchangeable."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


@dataclass
class ClusterRate:
    point: float
    lo: float
    hi: float
    n_episodes: int
    n_clusters: int
    naive_lo: float
    naive_hi: float
    boundary: bool = False   # k==0 or k==n: interval is Wilson, not bootstrapped

    @property
    def design_effect(self) -> float:
        """How much wider the cluster interval is. > 1.3 means task clustering is
        doing real work and the naive interval would have been misleading."""
        naive_w = self.naive_hi - self.naive_lo
        clust_w = self.hi - self.lo
        return clust_w / naive_w if naive_w > 0 else float("nan")

    def pretty(self, pct: bool = True) -> str:
        f = 100 if pct else 1
        s = "%" if pct else ""
        mark = "*" if self.boundary else ""
        return f"{self.point * f:.1f}{s} [{self.lo * f:.1f}, {self.hi * f:.1f}]{mark}"


def cluster_bootstrap_rate(
    items: list[tuple[str, bool]],
    n_boot: int = 4000,
    seed: int = 0,
    alpha: float = 0.05,
) -> ClusterRate:
    """Rate with a task-clustered bootstrap CI. ``items`` is [(task_id, hit), ...].

    Resamples tasks with replacement, keeping each sampled task's episodes intact.
    """
    if not items:
        return ClusterRate(*([float("nan")] * 3), 0, 0, float("nan"), float("nan"), True)

    by_task: dict[str, list[bool]] = defaultdict(list)
    for task, hit in items:
        by_task[task].append(hit)
    tasks = list(by_task)

    n = len(items)
    k = sum(1 for _, h in items if h)
    point, nlo, nhi = wilson(k, n)

    if len(tasks) < 2:
        return ClusterRate(point, nlo, nhi, n, len(tasks), nlo, nhi, True)

    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        num = den = 0
        for _ in range(len(tasks)):
            hits = by_task[tasks[rng.randrange(len(tasks))]]
            num += sum(hits)
            den += len(hits)
        if den:
            draws.append(num / den)
    draws.sort()
    lo = draws[int(alpha / 2 * len(draws))]
    hi = draws[int((1 - alpha / 2) * len(draws)) - 1]

    # Boundary degeneracy. When k == 0 (or k == n) every bootstrap resample also
    # yields 0 (or 1), so the percentile interval collapses to [0, 0] — an
    # interval that asserts the rate is known exactly from zero events. That is
    # the informed arm in this design, i.e. precisely the cell the headline
    # contrast rests on, so a collapsed interval there is not a cosmetic problem.
    # The nonparametric bootstrap simply has no information at the boundary;
    # fall back to the Wilson interval, which is defined there, and widen it by
    # the design effect observed in the other arm is NOT possible here, so we
    # take the (conservative) unclustered Wilson bound and flag it.
    boundary = (k == 0 or k == n)
    if boundary:
        lo, hi = nlo, nhi

    return ClusterRate(point, lo, hi, n, len(tasks), nlo, nhi, boundary)


def cluster_bootstrap_diff(
    a: list[tuple[str, bool]],
    b: list[tuple[str, bool]],
    n_boot: int = 4000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Difference a - b, resampling the SHARED task list so the two arms move
    together. This is the paired structure the design actually has: every task
    appears in both arms, so a task-level bootstrap that resampled each arm
    independently would throw away the pairing and inflate the interval.
    """
    ta: dict[str, list[bool]] = defaultdict(list)
    tb: dict[str, list[bool]] = defaultdict(list)
    for t, h in a:
        ta[t].append(h)
    for t, h in b:
        tb[t].append(h)
    shared = sorted(set(ta) & set(tb))
    if len(shared) < 2:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_clusters": len(shared), "p_boot": float("nan")}

    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        na = da = nb = db = 0
        for _ in range(len(shared)):
            t = shared[rng.randrange(len(shared))]
            na += sum(ta[t]); da += len(ta[t])
            nb += sum(tb[t]); db += len(tb[t])
        if da and db:
            draws.append(na / da - nb / db)
    draws.sort()
    point = (sum(1 for _, h in a if h) / len(a)) - (sum(1 for _, h in b if h) / len(b))
    lo = draws[int(alpha / 2 * len(draws))]
    hi = draws[int((1 - alpha / 2) * len(draws)) - 1]
    # two-sided bootstrap p: how often the resampled difference crosses zero
    p = 2 * min(
        sum(1 for d in draws if d <= 0) / len(draws),
        sum(1 for d in draws if d >= 0) / len(draws),
    )
    return {"diff": point, "lo": lo, "hi": hi, "n_clusters": len(shared),
            "p_boot": min(1.0, p)}


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Unclustered z-test. Report only alongside the clustered interval — at this
    N it will look more significant than the design supports."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    return z, 2 * (1 - _norm_cdf(abs(z)))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p. Exact and appropriate at small N — which is the
    regime this experiment is deliberately in."""
    def logfact(n: int) -> float:
        return math.lgamma(n + 1)

    def logp(a_, b_, c_, d_):
        n = a_ + b_ + c_ + d_
        return (logfact(a_ + b_) + logfact(c_ + d_) + logfact(a_ + c_) + logfact(b_ + d_)
                - logfact(n) - logfact(a_) - logfact(b_) - logfact(c_) - logfact(d_))

    obs = logp(a, b, c, d)
    r1, r2 = a + b, c + d
    c1 = a + c
    total = 0.0
    for i in range(max(0, c1 - r2), min(r1, c1) + 1):
        lp = logp(i, r1 - i, c1 - i, r2 - c1 + i)
        if lp <= obs + 1e-9:
            total += math.exp(lp)
    return min(1.0, total)


def cohens_kappa(a: list[str], b: list[str], categories: list[str]) -> float:
    """Carried over from ppa/pilot/reliability.py so disclosure agreement is
    computed the same way as in the sandbox corpus."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = {c: a.count(c) / n for c in categories}
    pb = {c: b.count(c) / n for c in categories}
    exp = sum(pa[c] * pb[c] for c in categories)
    return 1.0 if exp >= 1.0 else (obs - exp) / (1 - exp)
