SHELL := /bin/bash
COMPOSE := docker compose --env-file .env

.PHONY: help up down stop restart logs psql shell verify import seed scores reset backup restore

help:
	@echo "Toàn bộ stack:"
	@echo "  make up       - build + chạy postgres và api, chờ /health/ready"
	@echo "  make down     - dừng và xoá container (giữ volume)"
	@echo "  make restart  - khởi động lại api"
	@echo "  make logs     - log api"
	@echo "  make psql     - psql vào catalog"
	@echo "  make shell    - shell trong container postgres"
	@echo ""
	@echo "Dữ liệu:"
	@echo "  make import   - xoá sạch DB + nạp lại data/api_crawl_infor_mwgshop.json"
	@echo "  make seed     - sinh lại db/init/02_seed.sql từ JSON"
	@echo "  make scores   - sinh lại api/app/services/group_scores.py"
	@echo "  make verify   - kiểm tra toàn vẹn catalog"
	@echo "  make reset    - XOÁ volume rồi dựng lại DB từ db/init"
	@echo "  make backup   - dump ra ./backups/<timestamp>.dump"
	@echo "  make restore FILE=... - phục hồi từ dump"

up:
	@test -f .env || cp .env.example .env
	$(COMPOSE) up -d --build
	@./scripts/wait-for-db.sh
	@./api/scripts/wait-for-api.sh || true

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart api

logs:
	$(COMPOSE) logs -f api

psql: 
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-mwg} -d $${POSTGRES_DB:-mwgshop}

# Xoá sạch rồi nạp lại từ CSV — không cần xoá volume.
import:
	python3 scripts/import_catalog.py

seed:
	python3 scripts/gen_seed_sql.py

scores:
	python3 scripts/gen_group_scores.py

verify:
	./scripts/verify.sh

reset:
	$(COMPOSE) down -v
	$(MAKE) up

backup:
	@mkdir -p backups
	@ts=$$(date +%Y%m%dT%H%M%S); \
	$(COMPOSE) exec -T postgres pg_dump -U $${POSTGRES_USER:-mwg} -d $${POSTGRES_DB:-mwgshop} \
	  --format=custom --no-owner --no-acl > "backups/$$ts.dump"; \
	echo "backups/$$ts.dump"

restore:
	@test -n "$(FILE)" || { echo "usage: make restore FILE=backups/xxx.dump"; exit 1; }
	@$(COMPOSE) exec -T postgres pg_restore -U mwg -d mwgshop \
	  --clean --if-exists --no-owner --no-acl < "$(FILE)"
	@echo "restored $(FILE)"
