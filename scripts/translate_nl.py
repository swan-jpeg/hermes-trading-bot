#!/usr/bin/env python3
"""Detect and translate Dutch docstrings/comments into English.

Draait automatisch vóór elke push (via push-to-github.sh). Detecteert NL-regels
in docstrings/comments van alle .py-bestanden en vertaalt ze via de Nous API.
Blokkeert NIET — vertaalt gewoon en rapporteert wat er is aangepast.

Gebruik:
    uv run python scripts/translate_nl.py            # vertaal + rapporteer
    uv run python scripts/translate_nl.py --check   # alleen detecteren (exit 1 als NL)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Dutch words that almost never occur in English
# (reliable NL indicator — avoids false positives on "in/on/of/the").
NL_WORDS = re.compile(
    r"\b(de|het|een|voor|met|van|niet|geen|zijn|wordt|worden|draai|toon|maak|"
    r"gebruik|voer|zet|laat|zie|kies|maar|ook|nog|wel|hier|deze|dit|die|naar|"
    r"uit|aan|bij|door|onder|tussen|zonder|binnen|buiten|tegen|tot|weg|"
    r"zich|zelf|elkaar|ieder|elke|alles|niets|iets|heeft|hebben|moet|kunnen|"
    r"zullen|zou|waren|geen|niet|als|dan|en|op|van|met|voor|een|het|de|"
    r"worden|wordt|toont|na|er|zich|zelf|elkaar|ieder|elke|alles|niets|iets|"
    r"worden|wordt|heeft|hebben|moet|kunnen|zullen|zou|waren|geen|niet|"
    r"als|dan|en|op|van|met|voor|een|het|de)\b",
    re.IGNORECASE,
)

# Lines that are a docstring/comment (not functional code).
_COMMENT_RE = re.compile(r"^\s*(#|'''|\"\"\")")


def _get_agent_key() -> str | None:
    """Fetch the Nous agent_key from auth.json (without printing secrets)."""
    try:
        auth = json.loads((pathlib.Path.home() / ".hermes" / "auth.json").read_text())
        nous = auth["providers"]["nous"]
        return nous.get("agent_key") or nous.get("access_token")
    except Exception:  # noqa: BLE001
        return None


def _translate_text(text: str, key: str, base: str) -> str:
    """Translate one NL line into English via the Nous API."""
    payload = {
        "model": "qwen/qwen3-coder-flash",
        "messages": [
            {"role": "system", "content": (
                "You translate Dutch docstrings and code comments to English. "
                "Reply with ONLY the translation, no quotes, no explanation. "
                "Keep any code identifiers, variable names and symbols unchanged.")},
            {"role": "user", "content": f"Vertaal naar het Engels: {text}"},
        ],
        "max_tokens": 200,
        "temperature": 0.1,
    }
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    req.data = json.dumps(payload).encode()
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
        return d["choices"][0]["message"]["content"].strip()


def _find_nl_lines(path: pathlib.Path) -> list[tuple[int, str]]:
    """Find (line number, content) of NL docstrings/comments in a file."""
    hits: list[tuple[int, str]] = []
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return hits
    for i, line in enumerate(lines, 1):
        if _COMMENT_RE.match(line) and NL_WORDS.search(line):
            hits.append((i, line))
    return hits


def _rel(path: pathlib.Path) -> str:
    """Show a path relative to ROOT, or absolute if it lies outside."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _apply_translation(path: pathlib.Path, edits: list[tuple[int, str, str]]) -> None:
    """Apply translations to a file (line number -> new content)."""
    lines = path.read_text().splitlines()
    for lineno, _old, new in edits:
        if 1 <= lineno <= len(lines):
            # Preserve the indentation of the original line.
            indent = lines[lineno - 1][: len(lines[lineno - 1]) - len(lines[lineno - 1].lstrip())]
            lines[lineno - 1] = indent + new.strip()
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Vertaal NL-docstrings naar Engels")
    ap.add_argument("--check", action="store_true", help="alleen detecteren (exit 1 als NL)")
    ap.add_argument("--paths", nargs="*", default=None,
                    help="specifieke bestanden (default: alle .py)")
    args = ap.parse_args()

    if args.paths:
        files = [pathlib.Path(p) for p in args.paths]
    else:
        files = [p for p in ROOT.rglob("*.py")
                 if ".venv" not in p.parts and "node_modules" not in p.parts
                 and "var" not in p.parts]

    key = _get_agent_key()
    base = "https://inference-api.nousresearch.com/v1"
    if not key:
        print("⚠ Geen Nous agent_key gevonden in auth.json — kan niet vertalen.")
        return 2

    total_nl = 0
    total_translated = 0
    for path in sorted(files):
        hits = _find_nl_lines(path)
        if not hits:
            continue
        total_nl += len(hits)
        if args.check:
            print(f"NL: {_rel(path)} — {len(hits)} regels")
            continue
        # Translate each NL line.
        edits: list[tuple[int, str, str]] = []
        for lineno, line in hits:
            try:
                translated = _translate_text(line, key, base)
                edits.append((lineno, line, translated))
                total_translated += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠ vertaling mislukt {path.name}:{lineno}: {e}")
        if edits:
            _apply_translation(path, edits)
            print(f"✓ {_rel(path)} — {len(edits)} regels vertaald")

    if args.check:
        print(f"\n{total_nl} NL-regels gevonden in {len(files)} bestanden.")
        return 1 if total_nl else 0

    print(f"\nKlaar: {total_translated} NL-regels vertaald naar het Engels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
