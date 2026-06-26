import json
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def load_data(data_path="data/train.csv"):
    df = pd.read_csv(data_path)
    smiles_list = df["SMILES"].tolist()
    properties = df["property"].values
    return df, smiles_list, properties


def compute_descriptors(smiles_list, properties):
    rows = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows.append({
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "FracSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
            "Property": properties[i],
        })
    return pd.DataFrame(rows)


def plot_property_distribution(desc_df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(desc_df["Property"], bins=40, color="#2c7bb6", edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(desc_df["Property"].mean(), color="#d7191c", linestyle="--",
               linewidth=1.5, label=f"Mean = {desc_df['Property'].mean():.2f}")
    ax.axvline(desc_df["Property"].median(), color="#fdae61", linestyle=":",
               linewidth=1.5, label=f"Median = {desc_df['Property'].median():.2f}")
    ax.set_xlabel("Property (target)")
    ax.set_ylabel("Count")
    ax.set_title("Property Distribution — Training Set")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.boxplot(desc_df["Property"], vert=True, patch_artist=True,
               boxprops=dict(facecolor="#2c7bb6", alpha=0.6),
               medianprops=dict(color="#d7191c", linewidth=2),
               whiskerprops=dict(color="#333333"),
               capprops=dict(color="#333333"),
               flierprops=dict(marker="o", markerfacecolor="#d7191c",
                               markersize=4, alpha=0.5))
    stats = desc_df["Property"].describe()
    ax.text(1.15, stats["mean"], f"µ = {stats['mean']:.2f}\nσ = {stats['std']:.2f}\n"
            f"n = {int(stats['count'])}", transform=ax.transData,
            fontsize=11, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                      edgecolor="#cccccc"))
    ax.set_xticks([])
    ax.set_ylabel("Property (target)")
    ax.set_title("Property Summary Statistics")
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle("Dataset: Target Property Analysis", fontsize=16, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "property_distribution.png")
    return fig


def plot_molecular_descriptors(desc_df, save_path=None):
    descriptors = ["MolWt", "LogP", "HBD", "HBA", "RotBonds", "RingCount",
                   "HeavyAtoms"]
    labels = ["Molecular Weight (Da)", "LogP", "H-bond Donors",
              "H-bond Acceptors", "Rotatable Bonds", "Ring Count",
              "Heavy Atom Count"]
    colors = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a",
              "#b15928", "#a6cee3"]

    n = len(descriptors)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for i, (desc, label, color) in enumerate(zip(descriptors, labels, colors)):
        ax = axes[i]
        ax.hist(desc_df[desc], bins=35, color=color, edgecolor="white",
                linewidth=0.4, alpha=0.8)
        ax.axvline(desc_df[desc].median(), color="#333333", linestyle="--",
                   linewidth=1.2)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

        va = desc_df[desc].values
        ax.text(0.97, 0.93, f"µ={np.nanmean(va):.1f}  σ={np.nanstd(va):.1f}",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.8))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Molecular Descriptor Distributions", fontsize=16, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "molecular_descriptors.png")
    return fig


def plot_property_vs_descriptors(desc_df, save_path=None):
    descriptors = ["MolWt", "LogP", "HBD", "HeavyAtoms", "RingCount", "FracSP3"]
    labels = ["Molecular Weight (Da)", "LogP", "H-bond Donors",
              "Heavy Atom Count", "Ring Count", "Fraction sp³"]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(descriptors)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (desc, label, color) in enumerate(zip(descriptors, labels, colors)):
        ax = axes[i]
        x = desc_df[desc].values
        y = desc_df["Property"].values
        ax.scatter(x, y, s=12, c=[color], alpha=0.5, edgecolors="none")

        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 2:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            r2 = np.corrcoef(x[mask], y[mask])[0, 1] ** 2
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), color="#d7191c",
                    linewidth=1.5, linestyle="-", alpha=0.7)
            ax.text(0.97, 0.05, f"R² = {r2:.3f}", transform=ax.transAxes,
                    fontsize=10, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="none", alpha=0.8))

        ax.set_xlabel(label)
        ax.set_ylabel("Property")
        ax.grid(True, alpha=0.2)
        ax.set_title(f"Property vs {label.split('(')[0].strip()}",
                     fontsize=11)

    fig.suptitle("Property Correlation with Molecular Descriptors",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "property_vs_descriptors.png")
    return fig


def plot_smiles_length_distribution(smiles_list, save_path=None):
    lengths = [len(s) for s in smiles_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=40, color="#2c7bb6", edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(np.median(lengths), color="#d7191c", linestyle="--",
               linewidth=1.5, label=f"Median = {np.median(lengths):.0f}")
    ax.axvline(np.mean(lengths), color="#fdae61", linestyle=":",
               linewidth=1.5, label=f"Mean = {np.mean(lengths):.1f}")
    ax.set_xlabel("SMILES String Length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("SMILES Length Distribution — Training Set")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "smiles_length_distribution.png")
    return fig


def plot_training_curves(log_path, save_path=None):
    import json

    log_file = Path(log_path) / "metrics.json"
    if not log_file.exists():
        print(f"No metrics log found at {log_file}")
        return

    with open(log_file) as f:
        history = json.load(f)

    df = pd.DataFrame(history)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if "train_loss" in df.columns and "val_loss" in df.columns:
        ax.plot(df["epoch"], df["train_loss"], color="#2c7bb6", linewidth=1.5,
                label="Train", alpha=0.8)
        ax.plot(df["epoch"], df["val_loss"], color="#d7191c", linewidth=1.5,
                label="Validation", alpha=0.8)
        window = min(5, len(df) // 2)
        if window > 1:
            ax.plot(df["epoch"], df["train_loss"].rolling(window, center=True).mean(),
                    color="#2c7bb6", linewidth=1, linestyle="--", alpha=0.4)
            ax.plot(df["epoch"], df["val_loss"].rolling(window, center=True).mean(),
                    color="#d7191c", linewidth=1, linestyle="--", alpha=0.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
        ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    if "perplexity" in df.columns:
        ax.plot(df["epoch"], df["perplexity"], color="#6a3d9a", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.set_title("Validation Perplexity")
        ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    if "property_rmse" in df.columns:
        ax.plot(df["epoch"], df["property_rmse"], color="#33a02c", linewidth=1.5)
        ax.axhline(y=df["property_rmse"].min(), color="#333333", linestyle=":",
                   linewidth=1, alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Property RMSE")
        ax.set_title("Property Prediction RMSE")
        ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    if "lr" in df.columns:
        ax.plot(df["epoch"], df["lr"], color="#e31a1c", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Generator Training Dynamics", fontsize=16, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "training_curves.png")
    return fig


def plot_generative_metrics(log_path, save_path=None):
    import json

    log_file = Path(log_path) / "metrics.json"
    if not log_file.exists():
        print(f"No metrics log found at {log_file}")
        return

    with open(log_file) as f:
        history = json.load(f)

    df = pd.DataFrame(history)
    metric_cols = ["validity", "uniqueness", "novelty", "scaffold_diversity"]
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        print("No generative metrics found in log")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"validity": "#2c7bb6", "uniqueness": "#33a02c",
              "novelty": "#d7191c", "scaffold_diversity": "#6a3d9a"}
    markers = {"validity": "o", "uniqueness": "s", "novelty": "^",
               "scaffold_diversity": "D"}

    epoch_vals = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
    for col in available:
        valid_rows = df[col].notna()
        if valid_rows.sum() > 0:
            ax.plot(epoch_vals[valid_rows.values], df.loc[valid_rows, col],
                    color=colors.get(col, "#333333"), linewidth=1.5,
                    marker=markers.get(col, "o"), markersize=5,
                    label=col.replace("_", " ").title(), alpha=0.85)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Generative Quality Metrics Over Training")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "generative_metrics.png")
    return fig


def plot_smiles_heatmap(smiles_list, save_path=None):
    lengths = [len(s) for s in smiles_list]
    max_len = int(np.percentile(lengths, 95))
    tokens = set()
    for smi in smiles_list:
        for c in smi:
            tokens.add(c)
    token_list = sorted(tokens)
    token_to_idx = {t: i for i, t in enumerate(token_list)}

    n_samples = min(500, len(smiles_list))
    indices = np.random.RandomState(42).choice(len(smiles_list), n_samples, replace=False)

    matrix = np.zeros((n_samples, max_len), dtype=np.float32)
    for i, idx in enumerate(indices):
        smi = smiles_list[idx][:max_len]
        for j, c in enumerate(smi):
            matrix[i, j] = token_to_idx.get(c, 0)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix[:100], aspect="auto", cmap="viridis",
                   interpolation="nearest", vmin=0, vmax=len(token_list))
    ax.set_xlabel("Character Position")
    ax.set_ylabel("Molecule Index")
    ax.set_title("SMILES Character Occupation Map (100 molecules)")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Token ID (sorted alphabetically)", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "smiles_heatmap.png")
    return fig


def plot_embedding_projection(embeddings, properties, save_path=None):
    if embeddings is None:
        print("No embeddings provided — skipping projection plot")
        return

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=properties, cmap="RdYlBu_r",
                    s=15, alpha=0.6, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Molecule Embedding Space (PCA Projection)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("Property", fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "embedding_pca.png")
    return fig


def plot_correlation_matrix(desc_df, save_path=None):
    corr_cols = ["MolWt", "LogP", "HBD", "HBA", "RotBonds", "FracSP3",
                 "RingCount", "HeavyAtoms", "Property"]
    available = [c for c in corr_cols if c in desc_df.columns]
    corr = desc_df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest")

    labels = [c.replace("_", " ") for c in available]
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(available)):
        for j in range(len(available)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if abs(val) > 0.6 else "normal")

    ax.set_title("Feature Correlation Matrix", fontsize=15, pad=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Pearson R", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "correlation_matrix.png")
    return fig


def plot_scaffold_diversity(smiles_list, save_path=None):
    from collections import Counter
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            scaffolds.append(scaff)
        except Exception as e:
            import logging
            log = logging.getLogger(__name__)
            log.warning("Failed to compute Murcko scaffold: %s", e)
            continue

    counter = Counter(scaffolds)
    top_n = 15
    top_scaffolds = counter.most_common(top_n)
    names, counts = zip(*top_scaffolds) if top_scaffolds else ([], [])
    unique_scaffolds = len(counter)
    coverage = sum(counts) / max(len(scaffolds), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), counts, color="#2c7bb6", alpha=0.8,
                   edgecolor="white", linewidth=0.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9, color="#333333")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:40] + ("…" if len(n) > 40 else "") for n in names],
                       fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title(f"Top {top_n} Murcko Scaffolds\n"
                 f"{unique_scaffolds} unique scaffolds | "
                 f"Top {top_n} cover {coverage:.0%} of molecules",
                 fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "scaffold_diversity.png")
    return fig


def plot_ring_vs_property(desc_df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ring_counts = desc_df["RingCount"].value_counts().sort_index()
    ax.bar(ring_counts.index, ring_counts.values, color="#2c7bb6",
           edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Number of Rings")
    ax.set_ylabel("Count")
    ax.set_title("Ring Count Distribution")
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1]
    for rc in sorted(desc_df["RingCount"].unique()):
        subset = desc_df[desc_df["RingCount"] == rc]["Property"]
        ax.boxplot(subset, positions=[rc], widths=0.6,
                   patch_artist=True,
                   boxprops=dict(facecolor="#2c7bb6", alpha=0.5),
                   medianprops=dict(color="#d7191c", linewidth=2),
                   whiskerprops=dict(color="#333333"),
                   capprops=dict(color="#333333"),
                   flierprops=dict(marker="o", markersize=3, alpha=0.3))
    ax.set_xlabel("Number of Rings")
    ax.set_ylabel("Property")
    ax.set_title("Property vs Ring Count")
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Ring Structure Analysis", fontsize=16, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path / "ring_analysis.png")
    return fig


def resolve_run_dir(base_dir: str | Path, run_id: str | None = None) -> tuple[Path, str]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        existing = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("generate_")]
        numbers = []
        for name in existing:
            parts = name.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                numbers.append(int(parts[-1]))
        next_num = max(numbers) + 1 if numbers else 1
        run_id = f"generate_{next_num}"

    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, run_id


def generate_all_plots(data_path="data/train.csv", log_path=None,
                       embeddings=None, save_dir=None, run_id=None,
                       quiet: bool = False):
    df, smiles_list, properties = load_data(data_path)

    if save_dir is not None:
        run_dir, resolved_run_id = resolve_run_dir(save_dir, run_id=run_id)
        latest_dir = Path(save_dir) / "latest"
        run_id_str = resolved_run_id if run_id is None else run_id
    else:
        run_dir = None
        latest_dir = None
        run_id_str = run_id or "unnamed"

    desc_df = compute_descriptors(smiles_list, properties)

    def _save_to_all(fig, filename):
        for d in [run_dir, latest_dir]:
            if d is not None:
                fig.savefig(d / filename)
        plt.close(fig)

    _save_to_all(plot_property_distribution(desc_df), "property_distribution.png")
    _save_to_all(plot_molecular_descriptors(desc_df), "molecular_descriptors.png")
    _save_to_all(plot_correlation_matrix(desc_df), "correlation_matrix.png")
    _save_to_all(plot_property_vs_descriptors(desc_df), "property_vs_descriptors.png")
    _save_to_all(plot_smiles_length_distribution(smiles_list), "smiles_length_distribution.png")
    _save_to_all(plot_ring_vs_property(desc_df), "ring_analysis.png")
    _save_to_all(plot_scaffold_diversity(smiles_list), "scaffold_diversity.png")
    _save_to_all(plot_smiles_heatmap(smiles_list), "smiles_heatmap.png")

    if log_path:
        _save_to_all(plot_training_curves(log_path), "training_curves.png")
        _save_to_all(plot_generative_metrics(log_path), "generative_metrics.png")

    if embeddings is not None:
        _save_to_all(plot_embedding_projection(embeddings, properties), "embedding_pca.png")

    manifest = {
        "run_id": run_id_str,
        "timestamp": datetime.now().isoformat(),
        "data_path": str(data_path),
        "log_path": str(log_path) if log_path else None,
        "n_samples": len(smiles_list),
        "n_valid_smiles": len(desc_df),
    }
    for d in [run_dir, latest_dir]:
        if d is not None:
            with open(d / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

    if not quiet:
        if run_dir is not None:
            print(f"Plots saved to {run_dir}/ (synced to latest/)")
        else:
            print(f"Plots generated (run_id={run_id_str})")
    return run_id_str


def list_runs(base_dir: str | Path) -> list[dict]:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []
    runs = []
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name != "latest":
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    runs.append(json.load(f))
            else:
                runs.append({"run_id": d.name, "timestamp": "unknown"})
    return runs


if __name__ == "__main__":
    generate_all_plots(
        data_path="data/train.csv",
        log_path="outputs/checkpoints/generator_logs",
        save_dir="reports/figures/generator",
    )
    print("All plots generated.")
