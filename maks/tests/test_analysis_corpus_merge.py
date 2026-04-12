"""Corpus (bucket) is source of truth for stage-2 merge."""

import json
from pathlib import Path

import pandas as pd

from thesisdatarepo.analysis.config_loader import load_config
from thesisdatarepo.analysis.io_data import load_merged_frame


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_meta(path: Path, rows: list[dict], sep: str = ";") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, sep=sep, index=False, encoding="utf-8")


def test_nlp_txt_dir_filters_jsonl_to_existing_files(tmp_path: Path):
    """Optional nlp_txt_dir keeps only ids with a matching .txt on disk."""
    jl = tmp_path / "c.jsonl"
    _write_jsonl(
        jl,
        [
            {"id": "a_x", "text": "hello " * 20},
            {"id": "b_y", "text": "world " * 20},
        ],
    )
    txtdir = tmp_path / "nlp"
    txtdir.mkdir()
    (txtdir / "a_x.txt").write_text("hello", encoding="utf-8")
    # b_y.txt missing

    meta = tmp_path / "m.csv"
    _write_meta(
        meta,
        [
            {"member_id_ss": "a_x", "abstract_ts": "x", "Publisher": "P1", "year": "2020"},
            {"member_id_ss": "b_y", "abstract_ts": "y", "Publisher": "P2", "year": "2021"},
        ],
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        f"""
[paths]
corpus_jsonl = "{jl.as_posix()}"
metadata_csv = "{meta.as_posix()}"
output_dir = "{(tmp_path / 'out').as_posix()}"
nlp_txt_dir = "{txtdir.as_posix()}"

[embedding]
min_text_chars = 1
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    merged = load_merged_frame(cfg)
    assert len(merged) == 1
    assert merged["id"].iloc[0] == "a_x"


def test_bucket_drives_rows_and_drops_meta_only(tmp_path: Path):
    """Only ids in JSONL appear; metadata-only ids are excluded."""
    jl = tmp_path / "c.jsonl"
    _write_jsonl(
        jl,
        [
            {"id": "a", "text": "hello " * 20},
            {"id": "b", "text": "world " * 20},
        ],
    )
    meta = tmp_path / "m.csv"
    _write_meta(
        meta,
        [
            {"member_id_ss": "a", "abstract_ts": "x", "Publisher": "P1", "year": "2020"},
            {"member_id_ss": "c", "abstract_ts": "y", "Publisher": "P2", "year": "2021"},
        ],
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        f"""
[paths]
corpus_jsonl = "{jl.as_posix()}"
metadata_csv = "{meta.as_posix()}"
output_dir = "{(tmp_path / 'out').as_posix()}"

[embedding]
min_text_chars = 1
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    merged = load_merged_frame(cfg)
    assert len(merged) == 2
    assert set(merged["id"].tolist()) == {"a", "b"}
    assert "text_bucket" in merged.columns
    assert "hello" in merged.loc[merged["id"] == "a", "text_bucket"].iloc[0]
    row_b = merged.loc[merged["id"] == "b"].iloc[0]
    pub = row_b.get("Publisher", "")
    assert pd.isna(pub) or str(pub).strip() == "" or pub is None
