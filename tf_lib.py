"""Moved to ``sflu.plotting`` in Stage 7 of REFACTOR_PLAN.md; this alias keeps old imports working."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("sflu.plotting")
