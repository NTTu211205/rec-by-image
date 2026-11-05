#!/usr/bin/env bash
# Block until the postgres container reports healthy, or fail after ~120s.
set -euo pipefail
cd "$(dirname "$0")/.."

for i in $(seq 1 60); do
  status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' mwgshop-postgres 2>/dev/null || echo missing)
  case "$status" in
    healthy)
      echo "postgres is healthy"
      exit 0
      ;;
    missing|exited)
      echo "container not running (status=$status) - check: docker compose logs postgres" >&2
      exit 1
      ;;
  esac
  sleep 2
done

echo "timed out waiting for postgres health; last status=$status" >&2
exit 1
