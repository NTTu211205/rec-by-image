# Chạy local (không Docker). Docker: `docker compose up -d --build` từ thư mục gốc repo.
SHELL := /bin/bash
URL := http://127.0.0.1:$(shell grep -E '^API_PORT=' .env 2>/dev/null | cut -d= -f2 || echo 8000)

.PHONY: help test run health docs

help:
	@echo "Local (không Docker):"
	@echo "  make test      - pytest trong .venv"
	@echo "  make run       - uvicorn --reload ở cổng 8000"
	@echo "  make health    - gọi /health/ready"
	@echo "  make docs      - mở Swagger UI"
	@echo ""
	@echo "Docker (từ thư mục gốc repo):"
	@echo "  docker compose up -d --build   # postgres + api + adminer"

test:
	.venv/bin/python -m pytest tests -q

run:
	.venv/bin/uvicorn app.main:get_app --factory --reload --host 127.0.0.1 --port 8000

health:
	@curl -fsS "$(URL)/health/ready" | python3 -m json.tool || \
	  { echo "API chưa sẵn sàng"; exit 1; }

docs:
	@command -v open >/dev/null && open "$(URL)/docs" || echo "$(URL)/docs"
