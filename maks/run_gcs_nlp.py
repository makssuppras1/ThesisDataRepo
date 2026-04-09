"""Run GCS NLP from repo: ``uv run python maks/run_gcs_nlp.py`` (no package install needed)."""

from __future__ import annotations

import sys
from pathlib import Path

# maks/ must be on path so ``import thesisdatarepo`` works when this file is executed.
_MAKS = Path(__file__).resolve().parent
if str(_MAKS) not in sys.path:
    sys.path.insert(0, str(_MAKS))

from thesisdatarepo.cli_run_gcs import main

if __name__ == "__main__":
    raise SystemExit(main())
