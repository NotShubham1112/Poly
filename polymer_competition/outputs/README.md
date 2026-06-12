# Outputs Directory

Generated artifacts: checkpoints, logs, submissions, HPO trials.

| Subdirectory | Contents |
|---|---|
| `checkpoints/` | Per-fold model weights (.pt files) |
| `logs/` | TensorBoard event files, CSV training logs |
| `submissions/` | Final submission.csv and any ensemble variants |
| `hpo.db` | SQLite database for Optuna HPO trials |

All contents are .gitignored.
