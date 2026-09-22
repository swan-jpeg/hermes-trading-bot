#!/usr/bin/env bash
# Periodieke GitHub-update voor de trading-bot.
# Pusht nieuwe commits naar origin/master (gebruikt git credential helper of SSH).
# Logt output naar ~/.hermes/logs/git-push.log.
set -uo pipefail

REPO="${HOME}/trading-bot"
LOG="${HERMES_HOME:-$HOME/.hermes}/logs/git-push.log"
mkdir -p "$(dirname "$LOG")"

cd "$REPO" || { echo "repo niet gevonden: $REPO" >> "$LOG"; exit 1; }

# Update uit de remote halen (fast-forward) en eigen commits pushen.
{
  echo "=== $(date -Is) ==="
  git fetch origin 2>&1
  if ! git push origin master 2>&1; then
    echo "PUSH GEFAALD — check credentials (git credential helper / SSH / token)"
  fi
  echo
} >> "$LOG" 2>&1

# Schone exit-status (cron mag niet spammen bij niet-essentiële fout).
exit 0
