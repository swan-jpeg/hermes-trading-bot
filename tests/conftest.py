"""Shared test fixtures.

The pipeline's default closes_provider fetches live prices via yfinance.
Tests must stay offline and deterministic, so an autouse fixture replaces
the default with a deterministic stub for every test.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime

import numpy as np
import pytest

from hermes_bot.data.catalog import Catalog
from hermes_bot.data.graph import EconomicGraph
from hermes_bot.data.migrations import MigrationRunner
from hermes_bot.impact import ImpactRule
from hermes_bot.schemas import (
    Alias,
    AssetClass,
    CanonicalEntity,
    EntityType,
    Instrument,
    InstrumentType,
    RelationEdge,
    RelationType,
    ResolutionStatus,
    StrengthSemantics,
)


def _stub_closes(entity: str) -> np.ndarray:
    """Deterministic offline price series for any entity (no network)."""
    rng = np.random.default_rng(abs(hash(entity)) % 2**31)
    return np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100


@pytest.fixture(autouse=True)
def _offline_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the live price fetcher with a deterministic stub."""
    import hermes_bot.market.indicators_link as il
    monkeypatch.setattr(il, "fetch_closes", _stub_closes)


def _ent_id(name: str) -> str:
    return f"ent_{uuid.uuid5(uuid.NAMESPACE_DNS, f'{name}.test')}"


def _ins_id(name: str) -> str:
    return f"ins_{uuid.uuid5(uuid.NAMESPACE_DNS, f'{name}.test')}"


def _edge_id(name: str) -> str:
    return f"edg_{uuid.uuid5(uuid.NAMESPACE_DNS, f'{name}.test')}"


@pytest.fixture
def impact_catalog(tmp_path):
    """Fresh catalog with 10 synthetic entities, aliases, and relation edges."""
    db_path = tmp_path / "impact_catalog.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    MigrationRunner(conn).migrate()
    catalog = Catalog(conn)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    ids = {
        "nvda": _ent_id("nvda"),
        "amd": _ent_id("amd"),
        "asml": _ent_id("asml"),
        "microsoft": _ent_id("microsoft"),
        "apple": _ent_id("apple"),
        "apple_records": _ent_id("apple-records"),
        "us_gov": _ent_id("us-gov"),
        "eu": _ent_id("eu"),
        "spy": _ent_id("spy"),
        "semiconductor_sector": _ent_id("semiconductor-sector"),
    }

    entities = [
        (ids["nvda"], EntityType.COMPANY, "NVIDIA Corporation"),
        (ids["amd"], EntityType.COMPANY, "Advanced Micro Devices"),
        (ids["asml"], EntityType.COMPANY, "ASML Holding"),
        (ids["microsoft"], EntityType.COMPANY, "Microsoft Corporation"),
        (ids["apple"], EntityType.COMPANY, "Apple Inc"),
        (ids["apple_records"], EntityType.COMPANY, "Apple Records"),
        (ids["us_gov"], EntityType.GOVERNMENT, "United States Government"),
        (ids["eu"], EntityType.ORGANIZATION, "European Union"),
        (ids["spy"], EntityType.FUND, "SPDR S&P 500 ETF Trust"),
        (ids["semiconductor_sector"], EntityType.ORGANIZATION, "Semiconductor Sector"),
    ]
    for entity_id, entity_type, name in entities:
        catalog.entity.insert(
            CanonicalEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                canonical_name=name,
                created_at=now,
                valid_from=now,
            )
        )

    instruments = [
        (_ins_id("nvda"), ids["nvda"], AssetClass.STOCK, "NVIDIA Common Stock"),
        (_ins_id("amd"), ids["amd"], AssetClass.STOCK, "AMD Common Stock"),
        (_ins_id("spy"), ids["spy"], AssetClass.ETF, "SPY ETF"),
    ]
    for ins_id, issuer_id, asset_class, name in instruments:
        catalog.instrument.insert(
            Instrument(
                instrument_id=ins_id,
                issuer_entity_id=issuer_id,
                asset_class=asset_class,
                currency="USD",
                instrument_type=InstrumentType.COMMON_STOCK
                if asset_class == AssetClass.STOCK
                else InstrumentType.ETF_SHARE,
                canonical_name=name,
                created_at=now,
                valid_from=now,
            )
        )

    aliases = [
        ("NVDA", "ticker", ids["nvda"]),
        ("NVIDIA", "name", ids["nvda"]),
        ("AMD", "ticker", ids["amd"]),
        ("ASML", "ticker", ids["asml"]),
        ("MSFT", "ticker", ids["microsoft"]),
        ("Apple", "name", ids["apple"]),
        ("Apple", "name", ids["apple_records"]),
        ("SPY", "ticker", ids["spy"]),
        ("US Government", "name", ids["us_gov"]),
        ("European Union", "name", ids["eu"]),
    ]
    for alias, namespace, entity_id in aliases:
        status = (
            ResolutionStatus.AMBIGUOUS
            if alias == "Apple" and entity_id == ids["apple_records"]
            else ResolutionStatus.RESOLVED
        )
        catalog.alias.insert(
            Alias(
                alias=alias,
                namespace=namespace,
                entity_id=entity_id,
                resolution_status=status,
                created_at=now,
            )
        )

    edges = [
        ("nvda-sector", ids["nvda"], RelationType.MEMBER_OF_SECTOR, ids["semiconductor_sector"]),
        ("amd-sector", ids["amd"], RelationType.MEMBER_OF_SECTOR, ids["semiconductor_sector"]),
        ("asml-sector", ids["asml"], RelationType.MEMBER_OF_SECTOR, ids["semiconductor_sector"]),
        (
            "microsoft-sector",
            ids["microsoft"],
            RelationType.CUSTOMER_OF,
            ids["semiconductor_sector"],
        ),
        ("gov-sector", ids["us_gov"], RelationType.POLICY_APPLIES_TO, ids["semiconductor_sector"]),
        ("eu-sector", ids["eu"], RelationType.POLICY_APPLIES_TO, ids["semiconductor_sector"]),
        ("gov-spy", ids["us_gov"], RelationType.POLICY_APPLIES_TO, ids["spy"]),
    ]
    for name, from_id, rel, to_id in edges:
        catalog.relation.insert(
            RelationEdge(
                edge_id=_edge_id(name),
                from_entity_id=from_id,
                relation_type=rel,
                to_entity_id=to_id,
                available_at=now,
                evidence_ids=[f"ev_{name}"],
                source_ids=["src_test"],
                strength_semantics=StrengthSemantics.STRUCTURAL_DEPENDENCY,
                confidence=0.9,
                method_version="t08/v1",
                created_at=now,
            )
        )

    catalog.commit()

    rules = [
        ImpactRule(
            rule_id="rule_ai_chips",
            trigger_phrase="AI chips",
            required_phrases=("chips",),
            allowed_relation_types=(
                RelationType.MEMBER_OF_SECTOR,
                RelationType.CUSTOMER_OF,
                RelationType.POLICY_APPLIES_TO,
            ),
            direction="positive",
            impact_type="demand",
            confidence_label="high",
        ),
        ImpactRule(
            rule_id="rule_rate_hike",
            trigger_phrase="interest rate hike",
            allowed_relation_types=(RelationType.EXPOSED_TO, RelationType.POLICY_APPLIES_TO),
            direction="negative",
            impact_type="rates",
            confidence_label="medium",
        ),
        ImpactRule(
            rule_id="rule_rate_cut",
            trigger_phrase="interest rate cut",
            allowed_relation_types=(RelationType.EXPOSED_TO, RelationType.POLICY_APPLIES_TO),
            direction="positive",
            impact_type="rates",
            confidence_label="medium",
        ),
        ImpactRule(
            rule_id="rule_allegation",
            trigger_phrase="alleged",
            allowed_relation_types=(RelationType.MEMBER_OF_SECTOR, RelationType.SUBSIDIARY_OF),
            direction="unknown",
            impact_type="other",
            epistemic_type="allegation",
            confidence_label="low",
        ),
        ImpactRule(
            rule_id="rule_forecast",
            trigger_phrase="forecast",
            allowed_relation_types=(RelationType.MEMBER_OF_SECTOR, RelationType.EXPOSED_TO),
            direction="unknown",
            impact_type="other",
            epistemic_type="forecast",
            confidence_label="low",
        ),
    ]

    graph = EconomicGraph(conn)
    yield catalog, graph, rules, ids
    conn.close()
