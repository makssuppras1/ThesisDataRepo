"""CLI: load ``maks/nlp_gcs.toml`` and run :func:`process_gcs_prefix_nlp`."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from thesisdatarepo.gcs_nlp import process_gcs_prefix_nlp
from thesisdatarepo.pdf_context import FallbackPolicy


def _repo_root() -> Path:
    """Directory that contains ``pyproject.toml`` (prefers ``cwd`` when you run from the repo)."""
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file():
        return cwd
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file():
            return p
    return cwd


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GCS → NLP extraction using a TOML config (see maks/nlp_gcs.toml).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to TOML config (default: <repo>/maks/nlp_gcs.toml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and exit without calling GCS.",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    cfg_path = args.config if args.config is not None else root / "maks" / "nlp_gcs.toml"
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2

    data = _load_toml(cfg_path)
    gcs = data.get("gcs", {})
    paths = data.get("paths", {})
    run = data.get("run", {})
    pol = data.get("policy", {})

    bucket = str(gcs["bucket"])
    prefix = str(gcs["prefix"])
    txt_dir = root / paths["txt_dir"]
    manifest = root / paths["manifest"]
    jsonl_raw = paths.get("jsonl")
    jsonl = root / jsonl_raw if jsonl_raw else None

    max_pdfs = int(run.get("max_pdfs", -1))
    max_objects = None if max_pdfs < 0 else max_pdfs

    policy_type = str(pol.get("type", "keep_all")).lower()
    if policy_type == "keep_all":
        policy = FallbackPolicy.KEEP_ALL
    elif policy_type in ("keep_first_fraction", "first_fraction"):
        policy = FallbackPolicy.KEEP_FIRST_FRACTION
    else:
        print(f"Unknown policy.type: {policy_type}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"repo_root: {root}")
        print(f"config:    {cfg_path.resolve()}")
        print(f"bucket:    gs://{bucket}/{prefix}")
        print(f"txt_dir:   {txt_dir}")
        print(f"manifest:  {manifest}")
        print(f"jsonl:     {jsonl}")
        print(f"max_objects: {max_objects!r}")
        print(f"workers:   {run.get('max_workers', 6)}")
        return 0

    process_gcs_prefix_nlp(
        bucket,
        prefix,
        txt_dir,
        manifest,
        max_objects=max_objects,
        max_workers=int(run.get("max_workers", 6)),
        manifest_format="csv",
        jsonl_path=jsonl,
        clear_jsonl=bool(run.get("clear_jsonl", True)),
        resume=bool(run.get("resume", True)),
        force_fresh=bool(run.get("force_fresh", False)),
        min_page_fraction_back_matter=float(run.get("min_page_fraction_back_matter", 0.45)),
        policy=policy,
        fallback_fraction=float(pol.get("fallback_fraction", 0.95)),
        skip_if_empty=bool(run.get("skip_if_empty", True)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
