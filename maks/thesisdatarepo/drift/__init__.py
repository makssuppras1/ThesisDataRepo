"""Linguistic drift / AI-era contrast pipeline (metadata + NLP .txt corpus)."""

from thesisdatarepo.drift.cross_sectional import run_cross_sectional
from thesisdatarepo.drift.frames import load_modeling_frame
from thesisdatarepo.drift.its import fit_its, run_its_all_features
from thesisdatarepo.drift.jsd_drift import jsd_year_series

__all__ = [
    "fit_its",
    "jsd_year_series",
    "load_modeling_frame",
    "run_cross_sectional",
    "run_its_all_features",
]
