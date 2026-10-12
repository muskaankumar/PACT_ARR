.PHONY: help install test reproduce verify selftest clean

PY ?= python3

help:
	@echo "make install    editable install with dev + analysis extras"
	@echo "make test       run the harness test suite (30 tests)"
	@echo "make reproduce  regenerate results/ from data/"
	@echo "make verify     reproduce into a temp dir and diff against committed results/"
	@echo "make selftest   run the safety-critical detection selftest"
	@echo "make clean      remove caches and build artefacts"

install:
	$(PY) -m pip install -e ".[dev,analysis]"

test:
	PYTHONPATH=. $(PY) -m pytest tests/ -q

selftest:
	PYTHONPATH=. $(PY) -m ppa.pilot.detection_selftest

reproduce:
	cd analysis && $(PY) run_all.py --data ../data --out ../results

# Regenerate into a scratch dir and compare byte-for-byte with what is committed.
# Exits non-zero on any mismatch.
verify:
	@rm -rf .verify_out
	@cd analysis && $(PY) run_all.py --data ../data --out ../.verify_out >/dev/null
	@fail=0; \
	for f in $$(cd results && find . -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' \) ! -name 'README.md'); do \
	  cmp -s "results/$$f" ".verify_out/$$f" || { echo "MISMATCH: $$f"; fail=1; }; \
	done; \
	rm -rf .verify_out; \
	if [ $$fail -eq 0 ]; then echo "OK — committed results reproduce exactly from data/"; \
	else echo "FAILED — see mismatches above"; exit 1; fi

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache .ruff_cache .verify_out *.egg-info
