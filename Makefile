# ==========================
# Real-Time Crypto AI Service Makefile
# ==========================

# -------- VARIABLES --------
DOCKER_COMPOSE = docker compose
API_URL = http://localhost:8000

# -------- DOCKER COMMANDS --------
up:
	$(DOCKER_COMPOSE) up -d --build

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f model-server

restart:
	$(DOCKER_COMPOSE) restart model-server

# -------- TESTING --------
test:
	pytest -q --disable-warnings --maxfail=1

lint:
	ruff check .

format:
	black .

loadtest:
	python tests/load_test.py

# -------- PREDICT DEMO --------
predict:
	curl -s -X POST "$(API_URL)/predict" \
		-H "Content-Type: application/json" \
		-d '{"rows":[{"midprice":68000.5,"spread":1.2,"trade_intensity":14,"volatility_30s":0.02}]}'

# -------- UTIL --------
health:
	curl -s $(API_URL)/health
