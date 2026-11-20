"""Configuration loading and derived quantities."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_override(item: str) -> tuple[list[str], Any]:
    """'geometry.dx=60' -> (['geometry', 'dx'], 60)."""
    key, _, raw = item.partition("=")
    val = yaml.safe_load(raw)
    if isinstance(val, str):  # YAML 1.1 reads '1e-3' as a string
        try:
            val = float(val)
        except ValueError:
            pass
    return key.strip().split("."), val


class Config(dict):
    """Plain dict with attribute-style access to top-level sections and path helpers."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    @property
    def root(self) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(self["project_dir"]))).resolve()

    def path(self, rel: str | os.PathLike) -> Path:
        p = Path(os.path.expandvars(os.path.expanduser(str(rel))))
        return p if p.is_absolute() else self.root / p


def load_config(path=None, overrides: list[str] | None = None) -> Config:
    """config/default.yaml, then each YAML in `path` (one path or a list) merged in order,
    then `overrides` ('section.key=value')."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    paths = path if isinstance(path, (list, tuple)) else ([] if path is None else [path])
    for p in paths:
        if Path(p).resolve() == DEFAULT_CONFIG:
            continue
        with open(p) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        keys, val = _parse_override(item)
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    return Config(copy.deepcopy(cfg))


def save_config(cfg: Config, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
