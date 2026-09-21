#!/usr/bin/env bash
# Fix Obsidian op poort 9121 (KasmVNC "main bundle" crash).
#
# Reden fout: Obsidian draait als Electron in een Docker-container via VNC.
# De app-config (/root/.config/obsidian) is NIET gepersisteerd naar de host en
# bevat geen valide vault, waardoor de app in zijn main bundle crasht.
#
# Oplossing: verwijder corrupte workspace.json, zodat Obsidian fris opstart
# en de vault (/vaults = /home/olivier/obsidian-vault) opent.
#
# GEBRUIK (vereist root / docker-groep):
#   you@host$ sudo -i                          # of: sudo bash fix-obsidian.sh
#   you@host$ bash ~/trading-bot/scripts/fix-obsidian.sh
set -euo pipefail

# 1. Vind de obsidian container (bij voorkeur op naam 'obsidian').
CONTAINER="${OBSIDIAN_CONTAINER:-obsidian}"
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "Container '$CONTAINER' niet gevonden. Beschikbare containers:"
  docker ps --format '  - {{.Names}}  ({{.Image}})'
  echo "Zet juiste naam: OBSIDIAN_CONTAINER=<naam> bash $0"
  exit 1
fi

echo "==> Container '$CONTAINER' gevonden."

# 2. Toon huidige app-config (voor diagnose).
echo "==> Huidige Obsidian config in container:"
docker exec "$CONTAINER" sh -c 'ls -la /root/.config/obsidian/ 2>/dev/null | head' || echo "  (config map afwezig)"

# 3. Reset corrupte workspace / app staat. Backup eerst.
docker exec "$CONTAINER" sh -c '
  mkdir -p /root/.config/obsidian/_backup-$(date +%s);
  find /root/.config/obsidian -maxdepth 1 -type f -name "workspace*.json" -o -name "obsidian.json" | \
    xargs -r -I{} cp {} /root/.config/obsidian/_backup-$(date +%s)/ 2>/dev/null || true;
  rm -f /root/.config/obsidian/workspace.json /root/.config/obsidian/workspaces.json;
  echo "  workspace.json verwijderd (backup in _backup-<ts>)"'

# 4. Bevestig dat de vault zichtbaar is in de container.
echo "==> Vault-inhoud in container (/vaults):"
docker exec "$CONTAINER" sh -c 'ls /vaults/ 2>/dev/null && ls /vaults/TradingBot/ 2>/dev/null' || echo "  (vault map niet gevonden)"

# 5. Herstart de Obsidian-app zodat hij de vault opent.
echo "==> Obsidian-app herstarten..."
docker exec "$CONTAINER" sh -c 'pkill -f "bin/obsidian" 2>/dev/null; sleep 2; (sudo /usr/bin/obsidian --no-sandbox --disable-gpu >/dev/null 2>&1 &)' || echo "  (app handmatig herstarten via de container-shell)"

echo "==> Klaar. Herlaad http://localhost:9121 in je browser."
echo "TIP: blijft de Main-bundle-fout? Loop dan: docker restart $CONTAINER"
