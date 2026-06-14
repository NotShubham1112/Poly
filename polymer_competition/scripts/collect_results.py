"""
Collect training results from multiple sources (Colab accounts, laptops)
and merge them into one unified directory for ensemble inference.

Usage:
    python scripts/collect_results.py ^
        --sources ./run_A ./run_B ./run_C ^
        --output ./merged_results ^
        --person_tag true

What it does:
    - Copies all .pt checkpoints from each source's outputs/checkpoints/
    - Copies all .pkl predictions from each source's predictions/
    - Renames files with source prefix to avoid conflicts
    - Re-numbers folds if multiple sources have the same fold
    - Generates a combined config.yaml pointing to the merged dirs
"""
from __future__ import annotations

import argparse
import pickle
import re
import shutil
from pathlib import Path

import yaml


COPIED_FILES: list[Path] = []


def find_fold_number(path: Path) -> int | None:
    m = re.search(r"fold[_\s]*(\d+)", path.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def resolve_naming(
    source_name: str,
    rel_path: Path,
    dest_dir: Path,
    fold_map: dict[str, dict[int, int]],
) -> Path:
    stem = rel_path.stem
    ext = "".join(rel_path.suffixes)
    orig_fold = find_fold_number(rel_path)
    if orig_fold is not None:
        fold_remap = fold_map.setdefault(source_name, {})
        if orig_fold not in fold_remap:
            used = {v for m in fold_map.values() for v in m.values()}
            new_fold = 1
            while new_fold in used:
                new_fold += 1
            fold_remap[orig_fold] = new_fold
        mapped = fold_remap[orig_fold]
        new_stem = re.sub(
            r"fold[_\s]*\d+",
            f"fold_{mapped}",
            stem,
            flags=re.IGNORECASE,
        )
        if source_name not in new_stem:
            new_stem = f"{source_name}_{new_stem}"
    else:
        new_stem = f"{source_name}_{stem}"
    return dest_dir / f"{new_stem}{ext}"


def collect_dir(
    source_name: str,
    src: Path,
    dest: Path,
    fold_map: dict[str, dict[int, int]],
) -> int:
    count = 0
    for item in src.rglob("*"):
        if item.is_file() and item.suffix in {".pt", ".pkl", ".csv", ".json", ".yaml", ".yml", ".log"}:
            rel = item.relative_to(src)
            dest_path = resolve_naming(source_name, rel, dest, fold_map)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_path)
            COPIED_FILES.append(dest_path)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Merge training results from multiple sources"
    )
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="Source directories (each should contain outputs/ and predictions/)",
    )
    parser.add_argument(
        "--output", default="merged_results",
        help="Output directory (default: merged_results)",
    )
    parser.add_argument(
        "--person_tag", action="store_true",
        help="Use directory name as person tag instead of auto-detecting",
    )
    args = parser.parse_args()

    out = Path(args.output)
    ckpt_out = out / "outputs" / "checkpoints"
    pred_out = out / "predictions"
    ckpt_out.mkdir(parents=True, exist_ok=True)
    pred_out.mkdir(parents=True, exist_ok=True)

    fold_map: dict[str, dict[int, int]] = {}

    for src_path in args.sources:
        src = Path(src_path)
        if not src.exists():
            print(f"  [SKIP] {src} does not exist")
            continue

        name = src.name if args.person_tag else src.stem.replace(" ", "_")

        # Collect from outputs/checkpoints/
        ckpt_src = src / "outputs" / "checkpoints"
        if ckpt_src.exists():
            n = collect_dir(name, ckpt_src, ckpt_out, fold_map)
            print(f"  [{name}] checkpoints: {n} files")
        else:
            print(f"  [{name}] no checkpoints dir at {ckpt_src}")

        # Collect from predictions/
        pred_src = src / "predictions"
        if pred_src.exists():
            n = collect_dir(name, pred_src, pred_out, fold_map)
            print(f"  [{name}] predictions: {n} files")
        else:
            print(f"  [{name}] no predictions dir at {pred_src}")

        # Collect from reports/
        rep_src = src / "reports"
        rep_dest = out / "reports"
        if rep_src.exists():
            n = collect_dir(name, rep_src, rep_dest, fold_map)
            print(f"  [{name}] reports: {n} files")

    # Generate merged config
    merged_cfg = {
        "paths": {
            "checkpoints_dir": str(ckpt_out.resolve()),
            "predictions_dir": str(pred_out.resolve()),
            "submissions_dir": str((out / "submissions").resolve()),
        },
        "ensemble": {
            "strategy": "inverse_rmse",
        },
        "seed": 42,
    }
    cfg_path = out / "config_merged.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(merged_cfg, f, default_flow_style=False)

    print(f"\nMerged {len(COPIED_FILES)} files into {out}")
    print(f"Config: {cfg_path}")
    print(f"\nNext steps:")
    print(f"  python -m ensemble.build_ensemble --config {cfg_path}")
    if fold_map:
        print(f"\nFold remapping:")
        for src_name, mapping in fold_map.items():
            for orig, new in mapping.items():
                if orig != new:
                    print(f"  {src_name}: fold_{orig} -> fold_{new}")


if __name__ == "__main__":
    main()
