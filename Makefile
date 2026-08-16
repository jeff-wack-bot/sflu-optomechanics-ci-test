# Convenience targets. Everything runs with PYTHONHASHSEED pinned, because
# SFLU reduces graph nodes in set-iteration order and an unpinned seed shifts
# budgets by up to ~1e-3 between runs (REFACTOR_PLAN.md, Finding 1). The seed
# is read at interpreter startup, so it has to be set here rather than in
# pytest.ini.
export PYTHONHASHSEED := 0

PYTHON ?= python

.PHONY: help test guard baseline survey docs docs-quick serve clean-docs

help:
	@echo "test       run the test suite (reproducible)"
	@echo "guard      check model outputs against the stored baselines"
	@echo "baseline   re-record the baselines (deliberate; changes numbers)"
	@echo "survey     list modules that cannot be imported"
	@echo "docs       run the examples and rebuild the documentation"
	@echo "docs-quick rebuild documentation from existing tresults/ output"
	@echo "serve      preview the documentation locally"

test:
	$(PYTHON) -m pytest

guard:
	$(PYTHON) -m pytest tools/regression/test_regression.py

baseline:
	$(PYTHON) -m tools.regression.capture_baseline

survey:
	$(PYTHON) tools/regression/import_survey.py --quiet

docs:
	$(PYTHON) docs/generate_docs.py

docs-quick:
	$(PYTHON) docs/generate_docs.py --skip-tests

serve: docs-quick
	$(PYTHON) -m mkdocs serve -f docs/mkdocs.yml

clean-docs:
	rm -rf docs/_site docs/docs
