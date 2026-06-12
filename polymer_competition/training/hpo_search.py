"""
training.hpo_search.py
Optional Optuna integration for local hyperparameter sweeps.
"""
from __future__ import annotations

import argparse
import json

import optuna
import yaml
from pathlib import Path


def suggest_polychain(trial: optuna.Trial) -> dict:
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 384]),
        "n_backbone_layers": trial.suggest_int("n_backbone_layers", 3, 6),
        "n_hamf_layers": trial.suggest_int("n_hamf_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", default="polychain_hpo")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--storage", default="sqlite:///outputs/hpo.db")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(lambda t: objective(t), n_trials=args.n_trials)
    print("Best:", study.best_params, "value:", study.best_value)


def objective(trial: optuna.Trial) -> float:
    """Run a small training loop and return the validation RMSE."""
    # NOTE: implement by running training/train.py as a subprocess and
    # parsing the OOF RMSE from the saved .pkl file.
    raise NotImplementedError("Hook up to your training pipeline.")


if __name__ == "__main__":
    main()
