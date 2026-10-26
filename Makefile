.DEFAULT_GOAL := help
SHELL := /bin/bash
CONFIG ?= configs/base.yaml
UV := uv run

.PHONY: help install lint format typecheck test check ci-eval \
        data check-contamination generate verify train-notebook eval serve demo clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- setup & quality -------------------------------------------------------
install: ## Install the package and dev tools (exact pins from uv.lock)
	uv sync --all-extras --frozen || uv sync --all-extras

lint: ## Ruff lint + format check
	$(UV) ruff check .
	$(UV) ruff format --check .

format: ## Auto-format and fix lint issues
	$(UV) ruff format .
	$(UV) ruff check --fix .

typecheck: ## mypy (strict) on src/
	$(UV) mypy

test: ## Run the test suite
	$(UV) pytest

check: lint typecheck test ## Everything CI runs

# ---- pipeline (filled in phase by phase) ----------------------------------
data: ## Download Spider + BIRD Mini-Dev (~1 GB), extract schemas, build the registry
	$(UV) distill-sql data download --config $(CONFIG)
	$(UV) distill-sql data prepare --config $(CONFIG)

check-contamination: ## Fail if any eval DB/schema/question leaks into training data
	$(UV) distill-sql data check-contamination --config $(CONFIG)

generate: ## [Phase 2] Generate question->SQL pairs with teacher models
	@echo "Not implemented yet (Phase 2)"; exit 1

verify: ## [Phase 2] Execute and filter generated SQL, write stats report
	@echo "Not implemented yet (Phase 2)"; exit 1

train-notebook: ## [Phase 4] Print instructions for the Kaggle training notebook
	@echo "Not implemented yet (Phase 4)"; exit 1

eval: ## [Phase 3] Run the evaluation harness
	@echo "Not implemented yet (Phase 3)"; exit 1

ci-eval: ## [Phase 3] Re-score saved predictions on the fixed CI subset
	@echo "Not implemented yet (Phase 3)"; exit 1

serve: ## [Phase 7] Start the FastAPI service
	@echo "Not implemented yet (Phase 7)"; exit 1

demo: ## [Phase 9] Start the Gradio demo
	@echo "Not implemented yet (Phase 9)"; exit 1

clean: ## Remove caches (keeps data/ and runs/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
