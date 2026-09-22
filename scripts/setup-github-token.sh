#!/usr/bin/env bash
# GitHub-token instellen voor de trading-bot (veilig, geen shell-history).
#
# Gebruik:
#   1. Maak een Personal Access Token op https://github.com/settings/tokens
#      (klassiek token, scope: repo)
#   2. Draai dit script en plak het token wanneer gevraagd:
#        bash ~/trading-bot/scripts/setup-github-token.sh
#
# Het token wordt opgeslagen in ~/.git-credentials (alleen leesbaar door jou)
# en git gebruikt het voortaan automatisch voor pushes.
set -euo pipefail

REPO="${HOME}/trading-bot"
USERNAME="${1:-swan-jpeg}"

echo "=== GitHub-token instellen voor $REPO ==="
echo "Maak eerst een token op: https://github.com/settings/tokens"
echo "  - 'Generate new token (classic)'"
echo "  - Scope: 'repo' (volledige repo-toegang)"
echo

# Token lezen zonder echo (verborgen invoer).
read -r -s -p "Plak je GitHub Personal Access Token: " TOKEN
echo

if [ -z "$TOKEN" ]; then
  echo "Geen token ingevoerd — niets gewijzigd."
  exit 1
fi

# Credential helper 'store' inschakelen (persistent).
git config --global credential.helper store

# Token opslaan in ~/.git-credentials (beveiligde permissies).
umask 077
CRED_FILE="${HOME}/.git-credentials"
touch "$CRED_FILE"
# Verwijder eventueel oud github.com-rijtje en voeg nieuw toe.
grep -v "^https://${USERNAME}@" "$CRED_FILE" > "${CRED_FILE}.tmp" 2>/dev/null || true
echo "https://${USERNAME}:${TOKEN}@github.com" >> "${CRED_FILE}.tmp"
mv "${CRED_FILE}.tmp" "$CRED_FILE"
chmod 600 "$CRED_FILE"

echo
echo "Token opgeslagen. Test nu de push:"
echo "  bash ~/trading-bot/scripts/push-to-github.sh"
echo "  cat ~/.hermes/logs/git-push.log"
