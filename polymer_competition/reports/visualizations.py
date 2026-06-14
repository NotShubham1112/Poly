"""
reports.visualizations.py

Publication-quality Matplotlib plots for PolyChain evaluation.

Generates 8 plot types after every training run:
    1. Training curves (loss vs epoch)
    2. Actual vs Predicted scatter
    3. Residual plot
    4. Cross-validation RMSE bar chart
    5. Model comparison bar chart
    6. PolyChain ablation study
    7. Target property distribution
    8. HAMF attention heatmap

All plots saved to reports/plots/ as PNG (300 DPI).

Usage:
    from reports.visualizations import ReportGenerator
    gen = ReportGenerator("reports/plots")
    gen.plot_training_curves(train_losses, val_losses)
    gen.plot_pred_vs_actual(y_true, y_pred, model_name="PolyChain")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class ReportGenerator:
    """Generate and save all evaluation plots."""

    def __init__(self, output_dir: str | Path = "reports/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib for publication-quality output."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams.update({
                "figure.figsize": (8, 6),
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "lines.linewidth": 2,
                "lines.markersize": 6,
                "axes.grid": True,
                "grid.alpha": 0.3,
            })
        except Exception:
            pass

    def _get_fig(self, figsize=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize or (8, 6))
        return fig, ax

    def _save(self, fig, name: str):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Plot saved -> {path}")
        return path

    # ------------------------------------------------------------------
    # 1. Training Curves
    # ------------------------------------------------------------------
    def plot_training_curves(
        self,
        train_losses: list[float],
        val_losses: list[float],
        title: str = "Training Curves",
        save_name: str = "training_curve",
    ):
        """Plot train/val loss vs epoch."""
        import matplotlib.pyplot as plt

        fig, ax = self._get_fig()
        epochs = list(range(1, len(train_losses) + 1))
        ax.plot(epochs, train_losses, "o-", label="Train Loss", color="#4C78A8")
        ax.plot(epochs, val_losses, "s-", label="Validation Loss", color="#E45756")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(title)
        ax.legend()
        return self._save(fig, save_name)

    # ------------------------------------------------------------------
    # 2. Actual vs Predicted
    # ------------------------------------------------------------------
    def plot_pred_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "",
        save_name: str = "pred_vs_actual",
    ):
        """Scatter plot of predicted vs actual values."""
        import matplotlib.pyplot as plt

        fig, ax = self._get_fig()
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#4C78A8", edgecolors="white", linewidth=0.5)

        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        pad = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - pad, lims[1] + pad]
        ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect Prediction")

        # Compute metrics for annotation
        from scipy.stats import spearmanr
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
        sp = float(spearmanr(y_true, y_pred).correlation)
        ax.text(0.05, 0.95, f"RMSE={rmse:.4f}\nR2={r2:.4f}\nSpearman={sp:.4f}",
                transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        title = f"Predicted vs Actual"
        if model_name:
            title += f" ({model_name})"
        ax.set_title(title)
        ax.legend()
        return self._save(fig, save_name)

    # ------------------------------------------------------------------
    # 3. Residual Plot
    # ------------------------------------------------------------------
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "",
        save_name: str = "residuals",
    ):
        """Residual (y_true - y_pred) vs predicted values."""
        import matplotlib.pyplot as plt

        fig, ax = self._get_fig()
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=20, color="#4C78A8", edgecolors="white", linewidth=0.5)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Residual")

        # Mean absolute residual
        mar = np.mean(np.abs(residuals))
        ax.axhline(mar, color="orange", linestyle=":", linewidth=1, alpha=0.7, label=f"|MAR|={mar:.4f}")
        ax.axhline(-mar, color="orange", linestyle=":", linewidth=1, alpha=0.7)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (Actual - Predicted)")
        title = "Residual Plot"
        if model_name:
            title += f" ({model_name})"
        ax.set_title(title)
        ax.legend()
        return self._save(fig, save_name)

    # ------------------------------------------------------------------
    # 4. Cross-Validation RMSE
    # ------------------------------------------------------------------
    def plot_cv_rmse(
        self,
        fold_metrics: list[dict],
        model_name: str = "",
        save_name: str = "cv_rmse",
    ):
        """Bar chart of per-fold RMSE with mean/std annotation."""
        import matplotlib.pyplot as plt

        fig, ax = self._get_fig()
        folds = [f"Fold {m.get('fold', i)}" for i, m in enumerate(fold_metrics)]
        rmses = [m["rmse"] for m in fold_metrics]
        mean_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)

        colors = ["#4C78A8" if r <= mean_rmse else "#E45756" for r in rmses]
        bars = ax.bar(folds, rmses, color=colors, edgecolor="white", linewidth=1.5)

        # Mean line
        ax.axhline(mean_rmse, color="black", linestyle="--", linewidth=1.5,
                    label=f"Mean={mean_rmse:.4f} +/- {std_rmse:.4f}")

        # Value labels on bars
        for bar, rmse_val in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{rmse_val:.4f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Fold")
        ax.set_ylabel("RMSE")
        title = "Cross-Validation RMSE"
        if model_name:
            title += f" ({model_name})"
        ax.set_title(title)
        ax.legend()
        return self._save(fig, save_name)

    # ------------------------------------------------------------------
    # 5. Model Comparison
    # ------------------------------------------------------------------
    def plot_model_comparison(
        self,
        model_metrics: dict[str, float],
        metric_name: str = "rmse",
        save_name: str = "model_comparison",
    ):
        """Horizontal bar chart comparing models on a metric.

        Parameters
        ----------
        model_metrics : dict mapping model_name -> metric_value
        """
        import matplotlib.pyplot as plt

        fig, ax = self._get_fig(figsize=(10, max(4, len(model_metrics) * 0.6)))
        # Sort by metric value (lower is better for RMSE/MAE)
        sorted_items = sorted(model_metrics.items(), key=lambda x: x[1])
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
        bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=1)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel(metric_name.upper())
        ax.set_title(f"Model Comparison ({metric_name.upper()})")

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.001 * max(values),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", ha="left", va="center", fontsize=9)

        return self._save(fig, save_name)

    # ------------------------------------------------------------------
    # 6. Ablation Study
    # ------------------------------------------------------------------
    def plot_ablation(
        self,
        variant_metrics: dict[str, float],
        save_name: str = "ablation",
    ):
        """Bar chart showing ablation variants (Backbone, +HAMF, +PECGN, Full).

        Parameters
        ----------
        variant_metrics : dict mapping variant_name -> RMSE
        """
        import matplotlib.pyplot as plt

        fig, ax = self._get_fig(figsize=(8, 5))
        names = list(variant_metrics.keys())
        values = list(variant_metrics.values())

        # Color gradient: lighter = fewer components
        n = len(names)
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, n))

        bars = ax.bar(range(n), values, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("RMSE (lower is better)")
        ax.set_title("PolyChain Ablation Study")

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Highlight best
        best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

        plt.tight_layout()
        return self._save(fig, save_name)

    # ------------------------------------------------------------------
    # 7. Target Distribution
    # ------------------------------------------------------------------
    def plot_target_distribution(
        self,
        y: np.ndarray,
        target_name: str = "property",
        train_mask: Optional[np.ndarray] = None,
        val_mask: Optional[np.ndarray] = None,
        save_name: str = "target_distribution",
    ):
        """Histogram of target property values, optionally split by train/val."""
        import matplotlib.pyplot as plt

        fig, ax = self._get_fig()
        if train_mask is not None and val_mask is not None:
            ax.hist(y[train_mask], bins=40, alpha=0.6, label=f"Train (n={train_mask.sum()})",
                    color="#4C78A8", edgecolor="white")
            ax.hist(y[val_mask], bins=40, alpha=0.6, label=f"Val (n={val_mask.sum()})",
                    color="#E45756", edgecolor="white")
        else:
            ax.hist(y, bins=40, color="#4C78A8", edgecolor="white", alpha=0.8)

        ax.axvline(np.mean(y), color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean={np.mean(y):.3f}")
        ax.axvline(np.median(y), color="orange", linestyle=":", linewidth=1.5,
                    label=f"Median={np.median(y):.3f}")

        ax.set_xlabel(target_name)
        ax.set_ylabel("Count")
        ax.set_title(f"Target Distribution ({target_name})")
        ax.legend()
        return self._save(fig, save_name)

    # ------------------------------------------------------------------
    # 8. HAMF Attention Heatmap
    # ------------------------------------------------------------------
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        scale_labels: list[str] | None = None,
        layer_idx: int = 0,
        head_idx: int = 0,
        save_name: str = "hamf_attention",
    ):
        """Heatmap of HAMF cross-scale attention weights.

        Parameters
        ----------
        attention_weights : (n_layers, n_heads, n_scales, n_scales) or (n_scales, n_scales)
        """
        import matplotlib.pyplot as plt

        fig, ax = self._get_fig(figsize=(6, 5))

        if attention_weights.ndim == 4:
            attn = attention_weights[layer_idx, head_idx]
        elif attention_weights.ndim == 2:
            attn = attention_weights
        else:
            attn = attention_weights

        if scale_labels is None:
            scale_labels = ["Monomer", "Dimer", "Trimer"]

        im = ax.imshow(attn, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(scale_labels)))
        ax.set_yticks(range(len(scale_labels)))
        ax.set_xticklabels(scale_labels)
        ax.set_yticklabels(scale_labels)
        ax.set_xlabel("Key Scale")
        ax.set_ylabel("Query Scale")
        ax.set_title("HAMF Cross-Scale Attention")

        # Annotate cells
        for i in range(len(scale_labels)):
            for j in range(len(scale_labels)):
                ax.text(j, i, f"{attn[i, j]:.3f}", ha="center", va="center", fontsize=9)

        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ------------------------------------------------------------------
    # 9. Residual Distribution Histogram
    # ------------------------------------------------------------------
    def plot_residual_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "",
        save_name: str = "residual_distribution",
    ):
        """Histogram of residual values."""
        import matplotlib.pyplot as plt

        fig, ax = self._get_fig()
        residuals = y_true - y_pred
        ax.hist(residuals, bins=40, color="#4C78A8", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
        ax.axvline(np.mean(residuals), color="orange", linestyle=":", linewidth=1.5,
                    label=f"Mean={np.mean(residuals):.4f}")
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Count")
        title = "Residual Distribution"
        if model_name:
            title += f" ({model_name})"
        ax.set_title(title)
        ax.legend()
        return self._save(fig, save_name)

    # ------------------------------------------------------------------
    # 10. Batch Generate All Plots from Prediction Data
    # ------------------------------------------------------------------
    def generate_all_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "",
        train_losses: list[float] | None = None,
        val_losses: list[float] | None = None,
        fold_metrics: list[dict] | None = None,
        attention_weights: np.ndarray | None = None,
    ):
        """Generate all applicable plots from prediction data."""
        plots = []

        if train_losses and val_losses:
            plots.append(self.plot_training_curves(train_losses, val_losses,
                                                    title=f"Training Curves ({model_name})",
                                                    save_name=f"training_curve_{model_name}"))

        plots.append(self.plot_pred_vs_actual(y_true, y_pred, model_name,
                                               save_name=f"pred_vs_actual_{model_name}"))
        plots.append(self.plot_residuals(y_true, y_pred, model_name,
                                          save_name=f"residuals_{model_name}"))
        plots.append(self.plot_residual_distribution(y_true, y_pred, model_name,
                                                      save_name=f"residual_dist_{model_name}"))

        if fold_metrics:
            plots.append(self.plot_cv_rmse(fold_metrics, model_name,
                                            save_name=f"cv_rmse_{model_name}"))

        if attention_weights is not None:
            plots.append(self.plot_attention_heatmap(attention_weights,
                                                      save_name=f"hamf_attention_{model_name}"))

        return [p for p in plots if p is not None]


def save_summary_json(results: dict, output_path: str | Path):
    """Save results summary as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Summary saved -> {output_path}")
