"""In-memory correlation graph with the anti-double-counting rule (T18).

Edge capacity
-------------
Each node holds at most ``MAX_EDGES_PER_NODE`` (50) edges.  When adding an
edge would exceed the cap, edges are ranked by
``strength = abs(correlation_score) * certainty`` descending; ties break on
``edge_id`` ascending (lexicographic).  The weakest edge (lowest strength,
or highest ``edge_id`` on a tie) is dropped deterministically.

Anti-double-counting (core rule)
--------------------------------
``effective_correlation(a, b)`` returns ONLY the direct edge score between
``a`` and ``b`` (0.0 when no direct edge exists).  Path composition is
explicitly forbidden: no sums or products over hops, so edges A-B and B-C
never add anything to ``effective_correlation(A, C)``.

To correct for a shared neighbour, ``partial_correlation(a, c, control=[b])``
computes the correlation between the return series of ``a`` and ``c`` after
removing the linear effect of the control series: both series are
residualised on the control series via ordinary least squares (with
intercept), and Pearson correlation is computed over the residuals:

    a_res = a - (alpha_a + beta_a . b)      (OLS fit of a on controls)
    c_res = c - (alpha_c + beta_c . b)      (OLS fit of c on controls)
    partial = pearson(a_res, c_res)

Series must be registered with ``set_series(node_id, values)`` before
``partial_correlation`` is used.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from hermes_bot.correlation.schemas import CorrelationEdge, CorrelationNode

MAX_EDGES_PER_NODE = 50
PARTIAL_MIN_OBSERVATIONS = 3


def edge_strength(edge: CorrelationEdge) -> float:
    """Ranking strength: abs(weighted score) times certainty."""
    return abs(edge.correlation_score) * edge.certainty


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class CorrelationGraph:
    """Deterministic weighted correlation graph."""

    def __init__(self, max_edges_per_node: int = MAX_EDGES_PER_NODE) -> None:
        if max_edges_per_node < 1:
            raise ValueError("max_edges_per_node must be >= 1")
        self.max_edges_per_node = max_edges_per_node
        self._nodes: dict[str, CorrelationNode] = {}
        self._edges: dict[tuple[str, str], CorrelationEdge] = {}
        self._adjacency: dict[str, set[tuple[str, str]]] = {}
        self._series: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ nodes
    def add_node(self, node: CorrelationNode) -> None:
        self._nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, set())

    def get_node(self, node_id: str) -> CorrelationNode | None:
        return self._nodes.get(node_id)

    def nodes(self) -> list[str]:
        return sorted(self._nodes)

    def set_series(self, node_id: str, values: Sequence[float]) -> None:
        """Register the return series used for partial correlations."""
        self._series[node_id] = [float(v) for v in values]

    # ------------------------------------------------------------------ edges
    def add_edge(self, edge: CorrelationEdge) -> None:
        """Add (or replace) an edge, enforcing the per-node cap of 50."""
        for node_id in (edge.node_a, edge.node_b):
            if node_id not in self._nodes:
                raise KeyError(f"unknown node: {node_id}")
        key = _pair_key(edge.node_a, edge.node_b)
        old = self._edges.get(key)
        if old is not None and old.edge_id != edge.edge_id:
            # Replace the edge for this unordered pair.
            self._adjacency[old.node_a].discard(key)
            self._adjacency[old.node_b].discard(key)
        self._edges[key] = edge
        self._adjacency[edge.node_a].add(key)
        self._adjacency[edge.node_b].add(key)
        for node_id in (edge.node_a, edge.node_b):
            self._enforce_cap(node_id)

    def _enforce_cap(self, node_id: str) -> None:
        keys = self._adjacency[node_id]
        while len(keys) > self.max_edges_per_node:
            # Drop the weakest edge; ties break on highest edge_id (so the
            # deterministically "last" edge loses).
            ranked = sorted(
                keys,
                key=lambda k: (-edge_strength(self._edges[k]), self._edges[k].edge_id),
            )
            drop = ranked[-1]
            dropped_edge = self._edges.pop(drop)
            self._adjacency[dropped_edge.node_a].discard(drop)
            self._adjacency[dropped_edge.node_b].discard(drop)

    def get_edge(self, a: str, b: str) -> CorrelationEdge | None:
        return self._edges.get(_pair_key(a, b))

    def edges_for(self, node_id: str) -> list[CorrelationEdge]:
        """Edges touching ``node_id``, sorted by strength (desc), edge_id (asc)."""
        keys = self._adjacency.get(node_id, set())
        return [
            self._edges[k]
            for k in sorted(
                keys,
                key=lambda k: (-edge_strength(self._edges[k]), self._edges[k].edge_id),
            )
        ]

    # ------------------------------------------------- anti-double-counting
    def effective_correlation(self, a: str, b: str) -> float:
        """Direct edge score only; 0.0 without a direct edge.

        NEVER composes over paths: A-B and B-C contribute nothing here.
        """
        edge = self.get_edge(a, b)
        return edge.correlation_score if edge is not None else 0.0

    def partial_correlation(
        self, a: str, c: str, control: Sequence[str]
    ) -> float:
        """Pearson correlation of a and c after removing the linear effect
        of the control node series (OLS residualisation, deterministic).

        Returns 0.0 when series are missing or too short.
        """
        if not control:
            raise ValueError("at least one control node is required")
        try:
            xs = self._series[a]
            ys = self._series[c]
            controls = [self._series[name] for name in control]
        except KeyError:
            return 0.0
        n = min(len(xs), len(ys), *(len(cs) for cs in controls))
        if n < PARTIAL_MIN_OBSERVATIONS + len(controls):
            return 0.0
        xs = xs[:n]
        ys = ys[:n]
        controls = [cs[:n] for cs in controls]
        res_x = _ols_residuals(xs, controls)
        res_y = _ols_residuals(ys, controls)
        return max(-1.0, min(1.0, _pearson(res_x, res_y)))


def _ols_residuals(values: Sequence[float], controls: Sequence[Sequence[float]]) -> list[float]:
    """Residuals of ``values`` regressed on [1, controls...] via normal equations."""
    n = len(values)
    k = len(controls) + 1
    # Build design matrix columns: intercept + one per control.
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for t in range(n):
        row = [1.0] + [cs[t] for cs in controls]
        for i in range(k):
            xty[i] += row[i] * values[t]
            for j in range(k):
                xtx[i][j] += row[i] * row[j]
    beta = _solve(xtx, xty)
    return [
        values[t] - sum(beta[i] * ([1.0] + [cs[t] for cs in controls])[i] for i in range(k))
        for t in range(n)
    ]


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (deterministic)."""
    k = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return [0.0] * k  # singular: no linear effect removed
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(col + 1, k):
            factor = aug[r][col] / aug[col][col]
            for cidx in range(col, k + 1):
                aug[r][cidx] -= factor * aug[col][cidx]
    beta = [0.0] * k
    for col in range(k - 1, -1, -1):
        residual = sum(aug[col][j] * beta[j] for j in range(col + 1, k))
        beta[col] = (aug[col][k] - residual) / aug[col][col]
    return beta


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
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
