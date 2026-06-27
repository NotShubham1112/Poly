"""
training/scheduler.py
Experiment scheduler with GPU/CPU overlap, resume, and runtime budget enforcement.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class RunSpec:
    model_type: str
    target: str
    fold: int
    device: str = "cpu"
    priority: int = 5
    estimated_minutes: float = 10.0


class ExperimentScheduler:
    """Manages training job launch order with GPU/CPU overlap and resume."""

    DEVICE_MAP: dict[str, str] = {
        "ridge": "cpu", "xgb": "cpu", "lgb": "cpu",
        "catboost": "cpu", "rf": "cpu", "mlp": "cpu",
        "gcn": "cuda", "gat": "cuda", "graph_transformer": "cuda",
        "polychain": "cuda", "polychain_deep": "cuda",
        "polychain_wide": "cuda", "polychain_light": "cuda",
    }

    BATCHES: list[list[str]] = [
        ["ridge", "gcn"],
        ["xgb", "gat"],
        ["lgb", "graph_transformer"],
        ["catboost", "polychain"],
        ["rf", "polychain_deep"],
        ["mlp", "polychain_wide"],
        ["polychain_light"],
    ]

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.pred_dir = Path(self.cfg["paths"]["predictions_dir"])
        self.exp = self.cfg.get("experiment", {}).get("version", "v1")
        self.n_folds = self.cfg.get("cv", {}).get("n_folds", 5)
        self.targets = list(self.cfg.get("targets", {"tg": {}, "egc": {}}).keys())
        self.manifest_path = Path("experiments/manifest.json")
        self.completed_runs: set[tuple[str, str, int]] = set()
        self._load_manifest()

    def _load_manifest(self):
        if self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text())
            for entry in data:
                if entry.get("status") == "completed":
                    self.completed_runs.add(
                        (entry["model"], entry["target"], entry["fold"])
                    )
        print(f"Found {len(self.completed_runs)} completed runs in manifest")

    def get_pending_runs(self, targets: Optional[list[str]] = None,
                          model_types: Optional[list[str]] = None) -> list[RunSpec]:
        targets = targets or self.targets
        model_types = model_types or list(self.DEVICE_MAP.keys())
        runs = []
        for target in targets:
            for model_type in model_types:
                for fold in range(self.n_folds):
                    if (model_type, target, fold) in self.completed_runs:
                        continue
                    device = self.DEVICE_MAP.get(model_type, "cpu")
                    runs.append(RunSpec(
                        model_type=model_type, target=target,
                        fold=fold, device=device,
                    ))
        return runs

    def estimate_remaining_time(self, runs: list[RunSpec]) -> float:
        total = 0.0
        gpu_time, cpu_time = 0.0, 0.0
        for r in runs:
            est = r.estimated_minutes
            if "cuda" in r.device:
                gpu_time += est
            else:
                cpu_time += est
        return max(gpu_time, cpu_time)

    def budget_filter(self, runs: list[RunSpec],
                       remaining_minutes: float) -> list[RunSpec]:
        if remaining_minutes >= self.estimate_remaining_time(runs):
            return runs
        runs_sorted = sorted(runs, key=lambda r: r.priority)
        while runs_sorted and self.estimate_remaining_time(runs_sorted) > remaining_minutes:
            removed = runs_sorted.pop()
            print(f"  Budget: removing {removed.model_type}/{removed.target}/fold{removed.fold}")
        return runs_sorted

    def launch_run(self, run: RunSpec) -> bool:
        cuda_visible = ""
        if run.device.startswith("cuda:"):
            cuda_visible = f"CUDA_VISIBLE_DEVICES={run.device[-1]}"
        elif run.device == "cuda" and run.model_type in ("polychain", "polychain_deep"):
            cuda_visible = "CUDA_VISIBLE_DEVICES=0"

        cmd_parts = []
        if cuda_visible:
            cmd_parts.append(cuda_visible)
        cmd_parts.append(f"python -m training.train --model_type {run.model_type}")
        cmd_parts.append(f"--target {run.target} --fold {run.fold}")
        cmd_parts.append(f"--config config.yaml")

        cmd = " ".join(cmd_parts)
        print(f"  Launching: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED (code {result.returncode}): {result.stderr[:200]}")
            return False
        print(f"  Completed: {run.model_type}/{run.target}/fold{run.fold}")
        return True

    def run_all(self, targets: Optional[list[str]] = None,
                 model_types: Optional[list[str]] = None,
                 time_budget_minutes: Optional[float] = None):
        pending = self.get_pending_runs(targets, model_types)
        if not pending:
            print("All runs completed!")
            return

        print(f"Pending: {len(pending)} runs")

        if time_budget_minutes is not None:
            pending = self.budget_filter(pending, time_budget_minutes)
            print(f"After budget filter: {len(pending)} runs")

        for batch_idx, batch_models in enumerate(self.BATCHES):
            batch_runs = [r for r in pending if r.model_type in batch_models]
            if not batch_runs:
                continue
            print(f"\nBatch {batch_idx + 1} ({len(batch_runs)} runs): {batch_models}")

            for run in batch_runs:
                success = self.launch_run(run)
                if not success:
                    print(f"  WARNING: {run.model_type}/{run.target}/fold{run.fold} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--budget", type=float, default=None,
                        help="Time budget in minutes")
    args = parser.parse_args()

    scheduler = ExperimentScheduler(args.config)
    scheduler.run_all(
        targets=args.targets,
        model_types=args.models,
        time_budget_minutes=args.budget,
    )


if __name__ == "__main__":
    main()
