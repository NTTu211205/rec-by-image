#!/usr/bin/env bash
# Chờ API trả /health/ready = 200 (model đã nạp xong), không chỉ tiến trình sống.
set -euo pipefail

# Gọi từ thư mục gốc repo (Makefile) nên đọc .env ở đó.
PORT="${API_PORT:-8000}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  PORT="$(grep -E '^API_PORT=' "$ROOT/.env" | cut -d= -f2 | tr -d '[:space:]' || true)"
  PORT="${PORT:-8000}"
fi

URL="http://127.0.0.1:${PORT}/health/ready"
TIMEOUT="${HEALTH_TIMEOUT:-180}"
start=$SECONDS

printf 'Waiting for %s' "$URL"
while (( SECONDS - start < TIMEOUT )); do
  if curl -fsS --max-time 5 "$URL" >/dev/null 2>&1; then
    echo
    echo "✅ API sẵn sàng sau $(( SECONDS - start ))s"
    curl -fsS "$URL" | python3 -m json.tool 2>/dev/null || true
    exit 0
  fi
  printf '.'
  sleep 2
done

echo
echo "❌ API không sẵn sàng sau ${TIMEOUT}s — xem 'docker compose logs api'" >&2
exit 1
