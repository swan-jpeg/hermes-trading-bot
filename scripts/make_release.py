#!/usr/bin/env python3
"""Create a GitHub Release with download artifacts (core + full).

- Bepaalt de volgende versie (v0.1.0, v0.2.0, ...) uit de laatste tag.
- Bouwt twee zips:
    * core  — alleen de risk engine + kern (klein, snel te runnen)
    * full  — het hele project (alles, incl. heavy-model extras)
- Maakt een GitHub Release aan met beide artefacten (oude releases blijven).
- Werkt de README-versietabel bij.

Gebruik:
    uv run python scripts/make_release.py            # maak release (volgende versie)
    uv run python scripts/make_release.py --version v0.2.0   # specifieke versie
    uv run python scripts/make_release.py --dry-run  # alleen zips bouwen, geen release
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Core modules that go in the "core" zip (risk engine + backtest + schemas).
CORE_MODULES = [
    "hermes_bot/__init__.py",
    "hermes_bot/schemas.py",
    "hermes_bot/risk/",
    "hermes_bot/risk_v2/",
    "hermes_bot/risk_v2_1/",
    "hermes_bot/backtest/",
    "hermes_bot/simulation/",
    "hermes_bot/portfolio/",
    "hermes_bot/execution/",
    "hermes_bot/control_panel.py",
    "hermes_bot/config.py",
    "hermes_bot/cli.py",
    "config/",
    "tests/",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    ".env.example",
    ".gitignore",
    "install_windows.bat",
]

_SKIP_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", ".ruff_cache",
              "node_modules", "var", ".gitignore"}
_SKIP_EXT = {".pyc", ".db", ".sqlite"}


def _git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return r.stdout.strip()


def _get_agent_key() -> str | None:
    try:
        auth = json.loads((pathlib.Path.home() / ".hermes" / "auth.json").read_text())
        nous = auth["providers"]["nous"]
        return nous.get("agent_key") or nous.get("access_token")
    except Exception:  # noqa: BLE001
        return None


def _get_github_token() -> str | None:
    try:
        cred = pathlib.Path.home() / ".git-credentials"
        line = [ln for ln in cred.read_text().splitlines() if "github.com" in ln][0]
        return line.split(":", 2)[2].split("@")[0]
    except Exception:  # noqa: BLE001
        return None


def _next_version() -> str:
    """Determine the next version from the latest tag (v0.1.0 -> v0.2.0)."""
    tags = [t for t in _git("tag", "--list", "v*").splitlines() if t]
    if not tags:
        return "v0.1.0"
    latest = sorted(tags)[-1]
    m = re.match(r"v(\d+)\.(\d+)\.(\d+)", latest)
    if not m:
        return "v0.1.0"
    major, minor, patch = int(m[1]), int(m[2]), int(m[3])
    return f"v{major}.{minor}.{patch + 1}"


def _project_files(root: pathlib.Path, only: list[str] | None = None) -> list[pathlib.Path]:
    """Alle projectbestanden, gefilterd. `only` = subset (core)."""
    out: list[pathlib.Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if p.suffix.lower() in _SKIP_EXT:
            continue
        if rel.name in (".env",) or rel.name.endswith("_key.txt"):
            continue
        if only is not None:
            # Core: only files that fall under a core path.
            if not any(str(rel) == o or str(rel).startswith(o.rstrip("/") + "/")
                       for o in only):
                continue
        out.append(p)
    return out


def _build_zip(root: pathlib.Path, only: list[str] | None, version: str, kind: str) -> bytes:
    """Build a zip (bytes) of the project (core or full)."""
    buf = io.BytesIO()
    prefix = f"hermes-trading-bot-{version}-{kind}"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in _project_files(root, only):
            rel = p.relative_to(root)
            try:
                zf.write(p, arcname=f"{prefix}/{rel.as_posix()}")
            except OSError:
                continue
    buf.seek(0)
    return buf.getvalue()


def _upload_asset(release_id: str, token: str, path: pathlib.Path) -> None:
    """Upload an artifact to a release (GitHub uploads API)."""
    data = path.read_bytes()
    upload_base = (
        "https://uploads.github.com/repos/swan-jpeg/hermes-trading-bot/"
        f"releases/{release_id}/assets"
    )
    req = urllib.request.Request(
        f"{upload_base}?name={path.name}", method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/zip")
    req.data = data
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
        print(f"  ✓ geüpload: {d.get('name')} ({len(data)//1024} KB)")


def _update_readme(version: str, core_name: str, full_name: str) -> None:
    """Update the README version table (add a new row at the top)."""
    readme = ROOT / "README.md"
    text = readme.read_text()
    row = (f"| {version} | [core](https://github.com/swan-jpeg/hermes-trading-bot/"
           f"releases/download/{version}/{core_name}) | "
           f"[full](https://github.com/swan-jpeg/hermes-trading-bot/"
           f"releases/download/{version}/{full_name}) |")
    marker = "<!-- RELEASES -->"
    if marker in text:
        # Replace the table: new row right after the separator (before the marker).
        lines = text.splitlines()
        idx = next(i for i, ln in enumerate(lines) if marker in ln)
        # Find the separator before the marker (the table header).
        insert_at = None
        for i in range(idx - 1, max(0, idx - 6), -1):
            if lines[i].startswith("| ---"):
                insert_at = i + 1
                break
        if insert_at is None:
            insert_at = idx
        lines.insert(insert_at, row)
        readme.write_text("\n".join(lines) + "\n")
    else:
        # Add a Releases section before "## License".
        section = (
            f"## Releases\n\n"
            f"Download the latest version below. Each release keeps its own "
            f"downloads — old versions are never removed.\n\n"
            f"| Version | Core (risk engine) | Full (everything) |\n"
            f"| --- | --- | --- |\n"
            f"{row}\n\n"
            f"<!-- RELEASES -->\n\n"
        )
        text = text.replace("## License", section + "## License")
        readme.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description="Maak een GitHub Release met artefacten")
    ap.add_argument("--version", default=None, help="versie-tag (default: volgende)")
    ap.add_argument("--dry-run", action="store_true", help="alleen zips bouwen, geen release")
    args = ap.parse_args()

    version = args.version or _next_version()
    print(f"Versie: {version}")

    # Build the two zips.
    core_zip = _build_zip(ROOT, CORE_MODULES, version, "core")
    full_zip = _build_zip(ROOT, None, version, "full")
    core_name = f"hermes-trading-bot-{version}-core.zip"
    full_name = f"hermes-trading-bot-{version}-full.zip"
    print(f"  core: {len(core_zip)//1024} KB ({core_name})")
    print(f"  full: {len(full_zip)//1024} KB ({full_name})")

    if args.dry_run:
        print("Dry-run: zips gebouwd, geen release aangemaakt.")
        return 0

    token = _get_github_token()
    if not token:
        print("⚠ Geen GitHub-token gevonden in ~/.git-credentials.")
        return 2

    # Create the release.
    req = urllib.request.Request(
        "https://api.github.com/repos/swan-jpeg/hermes-trading-bot/releases",
        method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    req.data = json.dumps({
        "tag_name": version,
        "name": f"Hermes Trading Bot {version}",
        "body": ("Download **core** (alleen de risk engine + kern, klein) of "
                 "**full** (het hele project incl. heavy-model extras).\n\n"
                 "Zie de README voor installatie-instructies."),
        "draft": False,
        "prerelease": False,
    }).encode()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
            release_id = d["id"]
            print(f"Release aangemaakt: {d['html_url']}")
    except urllib.error.HTTPError as e:
        print(f"⚠ Release aanmaken mislukt: {e.code} {e.read()[:200]}")
        return 3

    # Upload beide artefacten.
    core_path = ROOT / "var" / "releases" / core_name
    full_path = ROOT / "var" / "releases" / full_name
    core_path.parent.mkdir(parents=True, exist_ok=True)
    core_path.write_bytes(core_zip)
    full_path.write_bytes(full_zip)
    _upload_asset(release_id, token, core_path)
    _upload_asset(release_id, token, full_path)

    # Update the README and commit.
    _update_readme(version, core_name, full_name)
    _git("add", "README.md")
    _git("-c", "user.email=bot@local", "-c", "user.name=bot",
         "commit", "-m", f"Release {version}: README download-tabel bijwerken")
    print("README bijgewerkt en gecommit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
