# Host-side convenience wrappers around the Dockerized dev environment.
# Run from the repo root on any platform that has Docker. See docs/DOCKER.md.
#
# Requires the sibling repos checked out next to this one:
#   ../rsscript  ../modern-c  ../tinygrad  ../portman

COMPOSE := docker compose
RUN := $(COMPOSE) run --rm dev

.DEFAULT_GOAL := help

.PHONY: help build shell bootstrap check fmt lint portman oracle clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Build the dev toolchain image
	$(COMPOSE) build dev

shell: build ## Open an interactive shell in the dev container
	$(RUN) bash

bootstrap: build ## Build the rss + mcc toolchains inside the container
	$(RUN) scripts/docker/bootstrap.sh

check: bootstrap ## rss check the port (expect 0 errors)
	$(RUN) bash -lc 'rss check tinygrad-rss'

fmt: bootstrap ## rss fmt over the port sources (stdout)
	$(RUN) bash -lc 'rss fmt tinygrad-rss/src/main.rss'

portman: bootstrap ## Run portman inventory + map + status
	$(RUN) bash -lc 'cd /work/portman && PYTHONPATH=src python3 -m portman inventory && PYTHONPATH=src python3 -m portman map && PYTHONPATH=src python3 -m portman status'

oracle: bootstrap ## Run the C-reference oracle roundtrip (numerical proof)
	$(RUN) bash -lc 'python3 oracle/roundtrip.py'

clean: ## Remove the container-local build volumes (forces a clean rebuild)
	$(COMPOSE) down -v
