from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Directory containing ``pyproject.toml`` (prefers cwd when run from the repo)."""
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file():
        return cwd
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file():
            return p
    return cwd


def resolve_path(p: str | Path, root: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()
