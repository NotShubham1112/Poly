from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_config_hash(cfg: dict) -> str:
    raw = yaml.dump(cfg, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


def load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def record_run(
    experiment: str,
    target: str,
    model_type: str,
    fold: int,
    score: float | None = None,
    checkpoint: str | None = None,
    duration_sec: int = 0,
    seed: int = 42,
    config_path: str = "config.yaml",
) -> None:
    manifest = load_manifest()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    record = {
        "experiment": experiment,
        "target": target,
        "model": model_type,
        "fold": fold,
        "status": "completed" if score is not None else "failed",
        "score": score,
        "checkpoint": checkpoint or "",
        "duration_sec": duration_sec,
        "seed": seed,
        "git_commit": get_git_commit(),
        "config_hash": get_config_hash(cfg),
        "environment": str(Path("experiments") / "environment.txt"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest.append(record)
    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
