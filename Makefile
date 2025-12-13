.DEFAULT_GOAL := help

SHELL := /bin/bash

# =========================== Конфигурация ===========================
PYTHON ?= python3
VENV_BIN := .venv/bin
PIP := $(VENV_BIN)/pip

PACKAGE_DIR := oasis
TEST_DIR := tests
SRC_PATHS := $(PACKAGE_DIR) $(TEST_DIR)

PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

DATA_DIR := data/deep_active_learning
STREAMLIT_APP := app.py
INDEX_SCRIPT := scripts/indexing.py
CQS_SCRIPT := scripts/gen_cqs.py
ONTOLOGY_SCRIPT := scripts/ontology.py
GEN_SCRIPT := scripts/generation.py
EVAL_SCRIPT := scripts/evaluation.py
HEALTH_SCRIPT := scripts/check_services_health.py

SERVICES_WAIT_SECONDS := 10
CLEAN_CACHES := .mypy_cache .pytest_cache .ruff_cache
MIN_PYTHON_VERSION := 3.11

.PHONY: help setup check-python check-env venv-check services-up services-down health-check bootstrap test lint type-check check format clean ingest index cqs ontology gen_e1 gen_e2 eval

# ============================ Справка ============================
help: ## Показать список целей Makefile и их описание
	@echo "Доступные цели:"
	@grep -E '^[[:alnum:]_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}{printf "  %-18s %s\n", $$1, $$2}'

# ========================== Бутстрап ==========================
check-python: ## Проверить версию Python (>=3.11)
	@echo "Проверка версии Python..."
	@$(PYTHON) -c "import sys; \
		v = sys.version_info; \
		min_v = tuple(map(int, '$(MIN_PYTHON_VERSION)'.split('.'))); \
		if (v.major, v.minor) < min_v[:2]: \
			print(f'Ошибка: требуется Python >= $(MIN_PYTHON_VERSION), найдена {v.major}.{v.minor}.{v.micro}'); \
			sys.exit(1) \
		else: \
			print(f'Python {v.major}.{v.minor}.{v.micro} ')"

check-env: ## Проверить наличие .env файла
	@if [ ! -f .env ]; then \
		if [ -f .env.sample ]; then \
			echo "Внимание: .env не найден, скопируйте .env.sample в .env и заполните значениями"; \
		else \
			echo "Ошибка: .env не найден и .env.sample отсутствует"; \
			exit 1; \
		fi \
	else \
		echo ".env найден "; \
	fi

venv-check: ## Проверить наличие виртуального окружения
	@if [ ! -d .venv ]; then \
		echo "Создание виртуального окружения..."; \
		$(PYTHON) -m venv .venv; \
		echo "Виртуальное окружение создано "; \
	else \
		echo "Виртуальное окружение найдено "; \
	fi

setup: venv-check check-python ## Установить зависимости проекта в локальное виртуальное окружение
	@echo "Установка зависимостей..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Зависимости установлены "

services-up: ## Запустить docker-compose службы в фоне
	@echo "Запуск Docker-сервисов..."
	docker compose up -d
	@echo "Ожидание запуска служб ($(SERVICES_WAIT_SECONDS) секунд)..."
	@sleep $(SERVICES_WAIT_SECONDS)
	@echo "Статус контейнеров:"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

services-down: ## Остановить docker-compose службы
	docker compose down

health-check: ## Проверить работоспособность зависимых сервисов
	$(PYTHON) $(HEALTH_SCRIPT)

bootstrap: check-python check-env ## Последовательно выполнить setup, services-up и health-check
	@echo "=========================================="
	@echo "Bootstrap окружения OASIS"
	@echo "=========================================="
	@$(MAKE) setup
	@echo ""
	@$(MAKE) services-up
	@echo ""
	@$(MAKE) health-check
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap завершен"
	@echo "=========================================="

# ========================= Разработка =========================
test: ## Запустить pytest с подробным выводом
	$(PYTEST) -v $(TEST_DIR)

test-cov: ## Запустить pytest с coverage отчетом
	$(PYTEST) --cov=oasis --cov=app --cov-report=term-missing --cov-report=html --cov-report=xml $(TEST_DIR)

test-cov-show: test-cov ## Запустить pytest с coverage и открыть HTML отчет
	@echo "Coverage отчет сохранён в htmlcov/index.html"

lint: ## Проверить стиль и качество кода через Ruff
	$(RUFF) check $(SRC_PATHS)

type-check: ## Прогнать статическую типизацию mypy
	$(MYPY) $(PACKAGE_DIR)

check: ## Последовательно выполнить lint, type-check и test
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test

format: ## Отформатировать код с помощью Ruff formatter
	$(RUFF) format $(SRC_PATHS)

run: ## Запустить Streamlit UI приложение
	$(PYTHON) -m streamlit run $(STREAMLIT_APP)

clean: ## Удалить байткод Python и кэши инструментов
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(CLEAN_CACHES)

# ========================= Pipeline =========================

