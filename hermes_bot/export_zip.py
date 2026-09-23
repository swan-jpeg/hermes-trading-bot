"""Build a downloadable zip of the project for GitHub export (also via iPad).

Slaat .venv, caches, var/ en .env over. Geeft op iPad de zip die je direct
naar GitHub-web (Upload files) of GitHub-mobile kunt pompen.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SKIP_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", ".ruff_cache",
              "node_modules", "var", ".gitignore"}
_SKIP_EXT = {".pyc", ".db", ".sqlite"}


def project_files(root: Path | None = None) -> list[Path]:
    """All project files, filtered — no venv/caches/secrets."""
    root = root or ROOT
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if p.suffix.lower() in _SKIP_EXT:
            continue
        # Never .env or credentials.
        if rel.name in (".env",) or rel.name.endswith("_key.txt"):
            continue
        out.append(p)
    return out


def build_zip_bytes(root: Path | None = None) -> bytes:
    """Create a .zip (bytes) of the project for download."""
    root = root or ROOT
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in project_files(root):
            rel = p.relative_to(root)
            try:
                zf.write(p, arcname=f"hermes-trading-bot/{rel.as_posix()}")
            except OSError:
                continue
    buf.seek(0)
    return buf.getvalue()


def build_github_instructions(user: str = "jouw-gebruikersnaam") -> str:
    """Short GitHub push instructions (also usable from iPad)."""
    return f"""# Hermes Trading Bot — naar GitHub zetten

## Option A — via this zip (works on iPad and any browser)
1. Download de zip hierboven (knop 'Download project (.zip)').
2. Ga naar https://github.com/new  → maak een nieuw publiek repo
   (naam: hermes-trading-bot).
3. Klik in de repo op **Add file → Upload files** en sleep de zip / de
   bestanden erin. (Op iPad kun je ook de GitHub-app gebruiken.)
4. Klik **Commit changes**.

## Option B — via git (on desktop / this server)
```bash
cd ~/trading-bot
git init -b main
git add -A && git commit -m "Initial Hermes trading bot"
git remote add origin https://github.com/{user}/hermes-trading-bot.git
git push -u origin main
```

## Afterwards (on your Windows PC or iPad)
- Download de zip / clone de repo.
- Windows: run `install_windows.bat` (vereist Python 3.11+ en uv).
- iPad: open in GitHub-app, of gebruik een git-client.
"""