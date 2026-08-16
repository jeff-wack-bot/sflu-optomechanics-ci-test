# Convenience targets. Everything runs with PYTHONHASHSEED pinned, because
# SFLU reduces graph nodes in set-iteration order and an unpinned seed shifts
# budgets by up to ~1e-3 between runs (REFACTOR_PLAN.md, Finding 1). The seed
# is read at interpreter startup, so it has to be set here rather than in
# pytest.ini.
export PYTHONHASHSEED := 0

PYTHON ?= python

.PHONY: help test guard guard-ci baseline survey docs docs-quick docs-strict \
        docs-site serve clean-docs

help:
	@echo "test        run the test suite (reproducible)"
	@echo "guard       check model outputs against the stored baselines (exact)"
	@echo "guard-ci    same, with cross-machine tolerances (what CI runs)"
	@echo "baseline    re-record the baselines (deliberate; changes numbers)"
	@echo "survey      list modules that cannot be imported"
	@echo "docs        run the examples and rebuild the documentation"
	@echo "docs-quick  rebuild documentation from existing tresults/ output"
	@echo "docs-strict rebuild, failing on missing examples or missing figures"
	@echo "docs-site   build the static site into docs/_site (as CI does)"
	@echo "serve       preview the documentation locally"

test:
	$(PYTHON) -m pytest

guard:
	$(PYTHON) -m pytest tools/regression/test_regression.py

# Cross-machine variant, for CI. The exact check above only holds on the
# machine that recorded the baseline: a different CPU or BLAS build moves most
# budgets by ~1e-8 and the worst-conditioned config by ~1e-3. These tolerances
# sit about a decade above the largest difference observed on a GitHub runner,
# so a real structural regression still fails while hardware noise does not.
# Topologies are strings and are compared exactly regardless.
GUARD_CI_RTOL ?= 1e-2
GUARD_CI_SCALE_ATOL ?= 1e-9

guard-ci:
	$(PYTHON) -m tools.regression.capture_baseline --check \
		--rtol $(GUARD_CI_RTOL) --scale-atol $(GUARD_CI_SCALE_ATOL)

baseline:
	$(PYTHON) -m tools.regression.capture_baseline

survey:
	$(PYTHON) tools/regression/import_survey.py --quiet

docs:
	$(PYTHON) docs/generate_docs.py

docs-quick:
	$(PYTHON) docs/generate_docs.py --skip-tests

# what CI runs: any documented example that vanishes, breaks, or stops
# producing figures fails the build instead of quietly dropping a page
docs-strict:
	$(PYTHON) docs/generate_docs.py --strict

docs-site: docs-strict
	$(PYTHON) -m mkdocs build -f docs/mkdocs.yml -d "$(CURDIR)/docs/_site"

serve: docs-quick
	$(PYTHON) -m mkdocs serve -f docs/mkdocs.yml

clean-docs:
	rm -rf docs/_site docs/docs
