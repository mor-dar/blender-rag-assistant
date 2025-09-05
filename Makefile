# Makefile for blender-rag-assistant
# Usage: `make help`

# ---- Config ----
PYTHON      ?= python
PIP         ?= pip
PERSIST_DIR ?= ./data/vector_db
COLLECTION  ?= blender_docs
K           ?= 5

# Prebuilt image as documented in the publication/README
DOCKER_IMAGE ?= mdar/blender-rag-assistant:v1.0.5

# ---- Phony ----
.PHONY: help setup web cli demo-db eval-demo docker-evaluate-web docker-build-full docker-run-web test clean

help: ## Show this help
	@echo "Targets:" ; \
	awk 'BEGIN {FS":.*?## "}; /^[a-zA-Z0-9_.-]+:.*?## /{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install Python dependencies
	$(PIP) install -r requirements.txt

web: ## Run Streamlit web UI locally
	$(PYTHON) main.py --web

cli: ## Run CLI locally (default mode)
	$(PYTHON) main.py

demo-db: ## Build the tiny demo DB (5 Blender 4.5 UI controls pages) and evaluate once
	$(PYTHON) scripts/evaluate_demo.py --autobuild-demo --persist_dir $(PERSIST_DIR) --collection $(COLLECTION) --k $(K)

eval-demo: ## Evaluate an existing local DB (writes outputs/eval_demo.*)
	$(PYTHON) scripts/evaluate_demo.py --persist_dir $(PERSIST_DIR) --collection $(COLLECTION) --k $(K)

docker-evaluate-web: ## One-line demo in Docker (builds demo DB + launches web)
	docker run --rm -p 8501:8501 -e GROQ_API_KEY=$$GROQ_API_KEY \
	  -v "$$(pwd)/data:/app/data" $(DOCKER_IMAGE) evaluate-web

docker-build-full: ## Build a full manual KB using the prebuilt image (persists to ./data)
	docker run --rm -v "$$(pwd)/data:/app/data" $(DOCKER_IMAGE) build-full

docker-run-web: ## Run the web UI against an existing KB (persisted in ./data)
	docker run --rm -p 8501:8501 -v "$$(pwd)/data:/app/data" $(DOCKER_IMAGE) run-web

test: ## Run tests (quiet)
	pytest -q

clean: ## Remove caches and local artifacts
	rm -rf __pycache__ .pytest_cache outputs/* $(PERSIST_DIR)/* 2>/dev/null || true

# Notes
# demo-db both builds the tiny demo DB and runs a quick eval so you immediately get outputs/eval_demo.md/json.
# If you prefer a strict separation, keep demo-db as-is and use eval-demo for re‑runs.
# The Docker targets mirror the commands already documented in your publication/README (prebuilt image, evaluate-web, build-full, run-web).
# No LLM key is needed for the evaluation script; the web/cli targets expect GROQ_API_KEY or OPENAI_API_KEY."