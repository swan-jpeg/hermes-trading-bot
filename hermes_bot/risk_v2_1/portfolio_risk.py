"""Whole-portfolio v2.1 risk approval (execution package, contracts §4.6).

Deterministic, portfolio-level hard-constraint enforcement for a complete
proposed target map. Long-only MVP. No execution wiring, no network, no state.

Design notes
------------
- All hard limits are read from ``config["risk"]``. A missing or non-finite
  hard limit NEVER falls back to a permissive default: the request is
  ``paused`` with an explicit ``missing_hard_limit``/``invalid_hard_limit``
  reason code.
- The result is order-independent. Aggregates are computed over the whole map
  with :func:`math.fsum` on sorted values, and reductions are a uniform
  proportional scale-down of every positive weight, so permuting the proposed
  list yields an identical approval.
- Reduction order (per spec): proportional scale-down of all positive proposed
  weights when gross exceeds the cap -> individual per-asset cap -> recheck the
  aggregate. If the recheck still fails the request is rejected.
- The returned approval is a frozen (immutable) :class:`RiskApprovalResult`
  carrying a canonical SHA-256 hash of the serialized request.

Reason codes are returned in the result; logging is never a substitute.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from hermes_bot.schemas import (
    RiskApprovalResult,
    RiskCheck,
    RiskRequest,
    RiskStatus,
)

# Hard limits that must be explicitly configured (no permissive defaults).
# max_sector_weight is required only when sector metadata is present.
REQUIRED_HARD_LIMITS: tuple[str, ...] = (
    "max_asset_weight",
    "max_gross_exposure",
    "max_leverage",
    "min_cash_weight",
    "max_position_count",
)

# --- Reason codes ----------------------------------------------------------
RC_INVALID_NAV = "invalid_nav"
RC_INVALID_CASH_WEIGHT = "invalid_cash_weight"
RC_INVALID_TIMESTAMP = "invalid_timestamp"
RC_UNKNOWN_INSTRUMENT = "unknown_instrument"
RC_DUPLICATE_INSTRUMENT = "duplicate_instrument"
RC_NON_FINITE_WEIGHT = "non_finite_weight"
RC_NEGATIVE_WEIGHT_SHORT = "negative_weight_short"
RC_WEIGHT_ABOVE_NAV = "weight_above_nav"
RC_UNSUPPORTED_SHORT_REQUEST = "unsupported_short_request"
RC_UNSUPPORTED_MARGIN_REQUEST = "unsupported_margin_request"
RC_MISSING_HARD_LIMIT = "missing_hard_limit"
RC_INVALID_HARD_LIMIT = "invalid_hard_limit"
RC_MISSING_PRICE = "missing_price"
RC_STALE_PRICE = "stale_price"
RC_MAX_POSITION_COUNT = "max_position_count_exceeded"
RC_MAX_ASSET_WEIGHT = "max_asset_weight_exceeded"
RC_MAX_GROSS_EXPOSURE = "max_gross_exposure_exceeded"
RC_MIN_CASH_WEIGHT = "min_cash_weight_exceeded"
RC_MAX_SECTOR_WEIGHT = "max_sector_weight_exceeded"
RC_CONSTRAINTS_UNSATISFIABLE = "constraints_unsatisfiable"

_EPS = 1e-9


def _is_finite(x: Any) -> bool:
    """True for a real finite number (bools are not numbers here)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _tz_aware(dt: datetime | None) -> bool:
    if not isinstance(dt, datetime):
        return False
    return dt.tzinfo is not None and dt.utcoffset() is not None


def request_hash(request: RiskRequest) -> str:
    """Canonical SHA-256 of the serialized request (incl. constraints version).

    ``mode="json"`` normalizes datetimes to ISO strings so the hash is stable
    regardless of internal representation. The proposed-position list and
    set-like sub-fields are order-normalized, so two requests that differ only
    by ordering hash identically and yield the same approval identity.
    """
    payload = request.model_dump(mode="json")

    positions = payload.get("proposed_positions") or []
    for p in positions:
        for key in ("input_record_ids", "reason_codes"):
            if p.get(key):
                p[key] = sorted(p[key])
    payload["proposed_positions"] = sorted(
        positions, key=lambda p: p.get("instrument_id") or ""
    )
    for key in ("reason_codes", "universe_ids"):
        if payload.get(key):
            payload[key] = sorted(payload[key])

    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def read_limits(cfg: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    """Extract required hard limits; report each missing/invalid one by name."""
    limits: dict[str, float] = {}
    bad: dict[str, str] = {}
    for name in REQUIRED_HARD_LIMITS:
        if name not in cfg or cfg[name] is None:
            bad[name] = "missing"
        elif not _is_finite(cfg[name]):
            bad[name] = "invalid"
        else:
            limits[name] = float(cfg[name])
    return limits, bad


def _zero_map(ids: list[str]) -> dict[str, float]:
    return {i: 0.0 for i in sorted(set(ids))}


def evaluate_portfolio(
    request: RiskRequest,
    cfg: dict[str, Any],
    *,
    clock: Callable[[], datetime] | None = None,
    ttl_seconds: float = 300.0,
) -> RiskApprovalResult:
    """Enforce all configured hard constraints on a complete target map."""
    clock = clock or (lambda: datetime.now(UTC))
    created_at = clock()
    rhash = request_hash(request)
    approval_id = f"rap_{rhash[:16]}"
    expires_at = created_at + timedelta(seconds=ttl_seconds)

    checks: list[RiskCheck] = []
    reason_codes: list[str] = []

    def add(
        constraint: str,
        passed: bool,
        *,
        observed: float | None = None,
        limit: float | None = None,
        rc: str = "",
    ) -> None:
        checks.append(
            RiskCheck(
                constraint=constraint,
                passed=passed,
                observed=observed,
                limit=limit,
                reason_code=rc,
            )
        )
        if rc and rc not in reason_codes:
            reason_codes.append(rc)

    positions = request.proposed_positions
    ids = [p.instrument_id for p in positions]

    def finish(
        status: RiskStatus,
        targets: dict[str, float] | None = None,
    ) -> RiskApprovalResult:
        return RiskApprovalResult(
            approval_id=approval_id,
            request_hash=rhash,
            status=status,
            approved_targets=targets if targets is not None else _zero_map(ids),
            checks=list(checks),
            reason_codes=list(reason_codes),
            created_at=created_at,
            expires_at=expires_at,
            constraints_version=request.constraints_version,
        )

    # ------------------------------------------------------------------
    # 1. Structural input validation -> rejected (no executable approval).
    # ------------------------------------------------------------------
    rejected = False

    def reject(cond: bool) -> None:
        nonlocal rejected
        rejected = rejected or cond

    ts_ok = _tz_aware(request.as_of)
    add("valid_as_of", ts_ok, rc="" if ts_ok else RC_INVALID_TIMESTAMP)
    reject(not ts_ok)

    nav_ok = _is_finite(request.nav) and request.nav > 0
    add(
        "positive_nav",
        nav_ok,
        observed=request.nav if _is_finite(request.nav) else None,
        rc="" if nav_ok else RC_INVALID_NAV,
    )
    reject(not nav_ok)

    cash_ok = _is_finite(request.cash_weight) and 0.0 <= request.cash_weight <= 1.0
    add(
        "valid_cash_weight",
        cash_ok,
        observed=request.cash_weight if _is_finite(request.cash_weight) else None,
        rc="" if cash_ok else RC_INVALID_CASH_WEIGHT,
    )
    reject(not cash_ok)

    want_short = bool(request.allow_short)
    add("long_only_request", not want_short,
        rc="" if not want_short else RC_UNSUPPORTED_SHORT_REQUEST)
    reject(want_short)

    want_margin = bool(request.allow_margin)
    add("no_margin_request", not want_margin,
        rc="" if not want_margin else RC_UNSUPPORTED_MARGIN_REQUEST)
    reject(want_margin)

    dup = len(ids) != len(set(ids))
    add("unique_instruments", not dup, rc="" if not dup else RC_DUPLICATE_INSTRUMENT)
    reject(dup)

    universe = set(request.universe_ids) if request.universe_ids is not None else None
    unknown = [
        i for i in ids
        if not isinstance(i, str) or not i.strip() or (universe is not None and i not in universe)
    ]
    cur_bad = any(
        (not isinstance(k, str)) or (not k.strip()) or (not _is_finite(v)) or v < 0
        for k, v in request.current_weights.items()
    )
    ids_ok = not unknown and not cur_bad
    add("known_instruments", ids_ok, rc="" if ids_ok else RC_UNKNOWN_INSTRUMENT)
    reject(not ids_ok)

    non_finite = [p.instrument_id for p in positions if not _is_finite(p.target_weight)]
    add("finite_weights", not non_finite, rc="" if not non_finite else RC_NON_FINITE_WEIGHT)
    reject(bool(non_finite))

    negative = [
        p.instrument_id for p in positions
        if _is_finite(p.target_weight) and p.target_weight < 0
    ]
    add("non_negative_weights", not negative,
        rc="" if not negative else RC_NEGATIVE_WEIGHT_SHORT)
    reject(bool(negative))

    above_nav = [
        p.instrument_id for p in positions
        if _is_finite(p.target_weight) and p.target_weight > 1.0 + _EPS
    ]
    add("weights_within_nav", not above_nav, rc="" if not above_nav else RC_WEIGHT_ABOVE_NAV)
    reject(bool(above_nav))

    pos_ts_bad = any(
        (p.price_available_at is not None and not _tz_aware(p.price_available_at))
        or (p.decision_time is not None and not _tz_aware(p.decision_time))
        for p in positions
    )
    add("valid_position_timestamps", not pos_ts_bad,
        rc="" if not pos_ts_bad else RC_INVALID_TIMESTAMP)
    reject(pos_ts_bad)

    if rejected:
        return finish(RiskStatus.REJECTED)

    # ------------------------------------------------------------------
    # 2. Hard-config validation -> paused (fail closed; no silent defaults).
    # ------------------------------------------------------------------
    paused = False
    limits, bad = read_limits(cfg)
    for name in REQUIRED_HARD_LIMITS:
        if name in bad:
            prefix = RC_MISSING_HARD_LIMIT if bad[name] == "missing" else RC_INVALID_HARD_LIMIT
            add(f"limit:{name}", False, rc=f"{prefix}:{name}")
            paused = True
        else:
            add(f"limit:{name}", True, limit=limits[name])

    sector_needed = any(p.sector_id for p in positions)
    if sector_needed:
        sw = cfg.get("max_sector_weight")
        if sw is None or not _is_finite(sw):
            add("limit:max_sector_weight", False, rc=f"{RC_MISSING_HARD_LIMIT}:max_sector_weight")
            paused = True
        else:
            limits["max_sector_weight"] = float(sw)
            add("limit:max_sector_weight", True, limit=float(sw))

    # ------------------------------------------------------------------
    # 3. Price freshness for required (positive-weight) positions -> paused.
    # ------------------------------------------------------------------
    required = [p for p in positions if _is_finite(p.target_weight) and p.target_weight > _EPS]
    stale_ids: list[str] = []
    missing_ids: list[str] = []
    for p in required:
        if p.is_price_fresh is False:
            stale_ids.append(p.instrument_id)
            continue
        if p.is_price_fresh is True:
            continue
        if p.price_available_at is None:
            missing_ids.append(p.instrument_id)
            continue
        if p.price_available_at > request.as_of:
            stale_ids.append(p.instrument_id)
            continue
        if request.max_stale_seconds is not None:
            age = (request.as_of - p.price_available_at).total_seconds()
            if age > request.max_stale_seconds + _EPS:
                stale_ids.append(p.instrument_id)

    if missing_ids:
        add("price_present", False, rc=RC_MISSING_PRICE)
        paused = True
    else:
        add("price_present", True)

    if stale_ids:
        add("price_fresh", False, rc=RC_STALE_PRICE)
        paused = True
    else:
        add("price_fresh", True)

    if paused:
        return finish(RiskStatus.PAUSED)

    # ------------------------------------------------------------------
    # 4. Deterministic constraint enforcement on the complete target map.
    # ------------------------------------------------------------------
    cap = min(
        limits["max_gross_exposure"],
        1.0 - limits["min_cash_weight"],
        1.0 + limits["max_leverage"],
        1.0,  # long-only MVP: gross can never exceed NAV
    )
    if cap < -_EPS:
        add("aggregate_feasible", False, observed=cap, limit=0.0,
            rc=RC_CONSTRAINTS_UNSATISFIABLE)
        return finish(RiskStatus.REJECTED)

    active = {
        p.instrument_id: p.target_weight
        for p in positions
        if _is_finite(p.target_weight) and p.target_weight > _EPS
    }
    position_count = len(active)
    count_ok = position_count <= limits["max_position_count"] + _EPS
    add("max_position_count", count_ok, observed=float(position_count),
        limit=limits["max_position_count"], rc="" if count_ok else RC_MAX_POSITION_COUNT)
    if not count_ok:
        # Dropping positions deterministically would invent a tie-break rule;
        # instead we fail closed and reject.
        return finish(RiskStatus.REJECTED)

    weights: dict[str, float] = {i: 0.0 for i in sorted(set(ids))}
    for iid, w in active.items():
        weights[iid] = w

    reduced = False
    gross0 = math.fsum(weights[i] for i in sorted(weights))
    if gross0 > cap + _EPS and gross0 > 0:
        scale = cap / gross0
        weights = {i: w * scale for i, w in weights.items()}
        reduced = True
        add("max_gross_exposure", True, observed=cap, limit=cap,
            rc=RC_MAX_GROSS_EXPOSURE)
    else:
        add("max_gross_exposure", True, observed=gross0, limit=cap)

    maw = limits["max_asset_weight"]
    clamped = False
    for i in sorted(weights):
        if weights[i] > maw + _EPS:
            weights[i] = maw
            clamped = True
    if clamped:
        reduced = True
        add("max_asset_weight", True, limit=maw, rc=RC_MAX_ASSET_WEIGHT)
    else:
        peak = max(weights.values()) if weights else 0.0
        add("max_asset_weight", True, observed=peak, limit=maw)

    gross1 = math.fsum(weights[i] for i in sorted(weights))
    if gross1 > cap + _EPS:
        add("aggregate_recheck", False, observed=gross1, limit=cap,
            rc=RC_CONSTRAINTS_UNSATISFIABLE)
        return finish(RiskStatus.REJECTED)
    add("aggregate_recheck", True, observed=gross1, limit=cap)

    cash1 = 1.0 - gross1
    cash_ok2 = cash1 + _EPS >= limits["min_cash_weight"]
    add("min_cash_weight", cash_ok2, observed=cash1, limit=limits["min_cash_weight"],
        rc="" if cash_ok2 else RC_MIN_CASH_WEIGHT)
    if not cash_ok2:
        return finish(RiskStatus.REJECTED)

    if sector_needed and "max_sector_weight" in limits:
        sector_totals: dict[str, float] = {}
        sec_of = {p.instrument_id: p.sector_id for p in positions}
        for i in sorted(weights):
            if weights[i] <= _EPS:
                continue
            sec = sec_of.get(i)
            if sec:
                sector_totals[sec] = sector_totals.get(sec, 0.0) + weights[i]
        worst = max(sector_totals.values()) if sector_totals else 0.0
        sec_ok = worst <= limits["max_sector_weight"] + _EPS
        add("max_sector_weight", sec_ok, observed=worst,
            limit=limits["max_sector_weight"], rc="" if sec_ok else RC_MAX_SECTOR_WEIGHT)
        if not sec_ok:
            # Sector cap cannot be fixed by uniform scaling without breaking
            # asset/gross logic; fail closed.
            return finish(RiskStatus.REJECTED)

    status = RiskStatus.REDUCED if reduced else RiskStatus.APPROVED
    targets = {i: round(weights[i], 12) for i in sorted(weights)}
    return finish(status, targets)