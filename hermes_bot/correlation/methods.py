"""Deterministic correlation methods (T18).

No training, no network.  Every method implements::

    score(series_a, series_b) -> CorrelationMethodResult

Rules shared by all methods:

* the score is clipped to [-1, 1];
* ``sample_size`` is the number of observations actually used;
* insufficient data returns score ``0.0`` explicitly, with a
  ``sample_size`` below the method's minimum (never silently omitted).

Methods
-------
RollingPearson(window=20)
    Pearson correlation over the last ``min(window, n)`` aligned
    observations.  Minimum 3 observations.

Spearman
    Pearson correlation over average ranks (ties get the mean rank).
    Minimum 3 observations.

KendallTau
    Kendall tau-b over all concordant/discordant pairs with tie
    correction: tau = (C - D) / sqrt((C + D + T_a) * (C + D + T_b)).
    Minimum 3 observations.

DynamicConditionalCorrelation
    EWMA (RiskMetrics-style, DCC(1,1)-like) dynamic correlation.
    Both series are demeaned over the full sample, then, with
    decay lambda = 0.94:

        var_a[t] = lambda * var_a[t-1] + (1 - lambda) * a[t] ** 2
        var_b[t] = lambda * var_b[t-1] + (1 - lambda) * b[t] ** 2
        cov[t]   = lambda * cov[t-1]   + (1 - lambda) * a[t] * b[t]
        rho[t]   = cov[t] / sqrt(var_a[t] * var_b[t])

    The returned score is rho at the final observation.  Minimum 5
    observations.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from hermes_bot.correlation.schemas import CorrelationMethodResult

MIN_OBSERVATIONS = 3
DCC_MIN_OBSERVATIONS = 5
EWMA_LAMBDA = 0.94


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _aligned(
    series_a: Sequence[float], series_b: Sequence[float]
) -> tuple[list[float], list[float]]:
    n = min(len(series_a), len(series_b))
    return list(series_a[:n]), list(series_b[:n])


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    var_x = sum(v * v for v in dx)
    var_y = sum(v * v for v in dy)
    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    cov = sum(a * b for a, b in zip(dx, dy, strict=True))
    return cov / math.sqrt(var_x * var_y)


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based); ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


class CorrelationMethod(Protocol):
    """Protocol for a deterministic correlation method."""

    name: str
    min_observations: int

    def score(
        self, series_a: Sequence[float], series_b: Sequence[float]
    ) -> CorrelationMethodResult: ...


class RollingPearson:
    """Pearson correlation over the most recent ``window`` observations."""

    name = "rolling_pearson"
    min_observations = MIN_OBSERVATIONS

    def __init__(self, window: int = 20) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window

    def score(
        self, series_a: Sequence[float], series_b: Sequence[float]
    ) -> CorrelationMethodResult:
        xs, ys = _aligned(series_a, series_b)
        k = min(len(xs), self.window)
        xs, ys = xs[-k:], ys[-k:]
        if k < self.min_observations:
            return CorrelationMethodResult(method=self.name, score=0.0, sample_size=k)
        return CorrelationMethodResult(
            method=self.name, score=_clip(_pearson(xs, ys)), sample_size=k
        )


class Spearman:
    """Spearman rank correlation (Pearson over average ranks)."""

    name = "spearman"
    min_observations = MIN_OBSERVATIONS

    def score(
        self, series_a: Sequence[float], series_b: Sequence[float]
    ) -> CorrelationMethodResult:
        xs, ys = _aligned(series_a, series_b)
        n = len(xs)
        if n < self.min_observations:
            return CorrelationMethodResult(method=self.name, score=0.0, sample_size=n)
        return CorrelationMethodResult(
            method=self.name,
            score=_clip(_pearson(_ranks(xs), _ranks(ys))),
            sample_size=n,
        )


class KendallTau:
    """Kendall tau-b rank correlation with tie correction."""

    name = "kendall_tau"
    min_observations = MIN_OBSERVATIONS

    def score(
        self, series_a: Sequence[float], series_b: Sequence[float]
    ) -> CorrelationMethodResult:
        xs, ys = _aligned(series_a, series_b)
        n = len(xs)
        if n < self.min_observations:
            return CorrelationMethodResult(method=self.name, score=0.0, sample_size=n)
        concordant = discordant = ties_a = ties_b = 0
        for i in range(n):
            for j in range(i + 1, n):
                da = xs[i] - xs[j]
                db = ys[i] - ys[j]
                if da == 0.0 and db == 0.0:
                    continue
                if da == 0.0:
                    ties_a += 1
                elif db == 0.0:
                    ties_b += 1
                elif da * db > 0.0:
                    concordant += 1
                else:
                    discordant += 1
        denom = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
        if denom == 0.0:
            return CorrelationMethodResult(method=self.name, score=0.0, sample_size=n)
        tau = (concordant - discordant) / denom
        return CorrelationMethodResult(method=self.name, score=_clip(tau), sample_size=n)


class DynamicConditionalCorrelation:
    """EWMA (RiskMetrics-style, DCC(1,1)-like) dynamic correlation.

    Deterministic: fixed decay ``lambda`` (default 0.94), series demeaned
    over the full sample.  See the module docstring for the exact recursion.
    """

    name = "dynamic_conditional"
    min_observations = DCC_MIN_OBSERVATIONS

    def __init__(self, decay: float = EWMA_LAMBDA) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay

    def score(
        self, series_a: Sequence[float], series_b: Sequence[float]
    ) -> CorrelationMethodResult:
        xs, ys = _aligned(series_a, series_b)
        n = len(xs)
        if n < self.min_observations:
            return CorrelationMethodResult(method=self.name, score=0.0, sample_size=n)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        dx = [x - mean_x for x in xs]
        dy = [y - mean_y for y in ys]
        lam = self.decay
        var_a = dx[0] * dx[0]
        var_b = dy[0] * dy[0]
        cov = dx[0] * dy[0]
        rho = 0.0
        for t in range(1, n):
            var_a = lam * var_a + (1 - lam) * dx[t] * dx[t]
            var_b = lam * var_b + (1 - lam) * dy[t] * dy[t]
            cov = lam * cov + (1 - lam) * dx[t] * dy[t]
            denom = math.sqrt(var_a * var_b)
            rho = cov / denom if denom > 0.0 else 0.0
        return CorrelationMethodResult(method=self.name, score=_clip(rho), sample_size=n)


def default_methods(window: int = 20) -> dict[str, CorrelationMethod]:
    """The four standard methods, keyed by method name."""
    methods: list[CorrelationMethod] = [
        RollingPearson(window=window),
        Spearman(),
        KendallTau(),
        DynamicConditionalCorrelation(),
    ]
    return {m.name: m for m in methods}
