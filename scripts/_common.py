from __future__ import annotations

from pathlib import Path
from typing import Any

from alphaforge.utils import load_yaml, save_json, timestamp_id


def make_run_dir(runs_dir: str | Path = "runs", prefix: str = "run") -> Path:
    run_dir = Path(runs_dir) / timestamp_id(prefix)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_latest(run_dir: Path) -> None:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir.resolve()))


def latest_run_dir(runs_dir: str | Path = "runs") -> Path:
    pointer = Path(runs_dir) / "latest_run.txt"
    if not pointer.exists():
        raise FileNotFoundError("no latest run found; run scripts/run_walk_forward.py first")
    run_dir = Path(pointer.read_text().strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"latest run directory does not exist: {run_dir}")
    return run_dir


def load_configs(
    model_config: str | Path = "configs/models.yaml",
    data_config: str | Path = "configs/data.yaml",
    feature_config: str | Path = "configs/features.yaml",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return load_yaml(model_config), load_yaml(data_config), load_yaml(feature_config)


def configure_fast_demo(model_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> None:
    data_cfg["source"] = "synthetic"
    data_cfg.setdefault("synthetic", {})
    data_cfg["synthetic"].update({"n_symbols": 8, "n_days": 420, "seed": 42})
    model_cfg["models"] = [
        {"name": "zero_baseline"},
        {"name": "momentum_baseline", "params": {"feature": "momentum_20", "scale": 0.05}},
        {"name": "ridge", "params": {"alpha": 10.0}},
    ]
    model_cfg.setdefault("walk_forward", {})
    model_cfg["walk_forward"].update(
        {
            "min_train_days": 180,
            "test_days": 40,
            "step_days": 40,
            "embargo_days": max(model_cfg.get("horizons", [20])),
            "max_windows": 2,
        }
    )


def save_meta(run_dir: Path, **meta: Any) -> None:
    save_json(meta, run_dir / "run_meta.json")
