from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from models.generator import (
    SELFIESTokenizer,
    GeneratorConfig,
    GraphEncoder,
    GeneratorDecoder,
    GenerativeLoss,
    SELFIESMask,
    MoleculeValidator,
    CurriculumScheduler,
    GenerativeMetrics,
)
from training.train_utils import (
    set_seed, save_checkpoint, load_checkpoint,
    MetricTracker,
)

DECODE_CONFIG = {
    "temperature": 1.0,
    "top_k": 40,
    "top_p": 0.9,
    "max_len": 80,
}


class PropertyScaler:
    def __init__(self):
        self.mean: float = 0.0
        self.std: float = 1.0
        self.fitted: bool = False

    def fit(self, values: list[float]) -> None:
        arr = np.array(values, dtype=np.float64)
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr)) + 1e-8
        self.fitted = True

    def transform(self, values: list[float] | np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float32)
        return (v - self.mean) / self.std

    def inverse(self, values: list[float] | np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float32)
        return v * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "fitted": self.fitted}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.std = state["std"]
        self.fitted = state["fitted"]


class SMILESDataset(Dataset):
    def __init__(
        self,
        smiles_list: list[str],
        tokenizer: SELFIESTokenizer,
        properties: list[float] | None = None,
        max_len: int = 256,
        build_graphs: bool = False,
    ):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.records: list[dict] = []

        for i, smi in enumerate(smiles_list):
            tokens = tokenizer.try_encode(smi)
            if tokens is None:
                continue

            if len(tokens) > max_len:
                tokens = tokens[: max_len - 1]
                tokens = torch.cat([tokens, torch.LongTensor([tokenizer.eos_token_id])])

            rec = {
                "input_ids": tokens,
                "target_ids": tokens.clone(),
                "smiles": smi.strip(),
            }
            if properties is not None:
                rec["property"] = torch.tensor(properties[i], dtype=torch.float)

            if build_graphs:
                from features.graph_utils import build_multiscale
                graph_sample = build_multiscale(smi.strip())
                if graph_sample is None:
                    continue
                rec["graph_sample"] = graph_sample

            self.records.append(rec)

        if not self.records:
            fallback = tokenizer.try_encode("CCO")
            if fallback is not None:
                self.records.append({
                    "input_ids": fallback,
                    "target_ids": fallback.clone(),
                    "smiles": "CCO",
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def collate_generator(batch: list[dict]) -> dict:
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    max_seq = max(t.size(0) for t in input_ids)
    padded_input = torch.full((len(batch), max_seq), 0, dtype=torch.long)
    padded_target = torch.full((len(batch), max_seq), 0, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(input_ids, target_ids)):
        l = inp.size(0)
        padded_input[i, :l] = inp
        padded_target[i, :l] = tgt

    result = {
        "input_ids": padded_input,
        "target_ids": padded_target,
        "smiles": [item["smiles"] for item in batch],
    }
    if "property" in batch[0]:
        result["property"] = torch.stack([item["property"] for item in batch])

    if "graph_sample" in batch[0]:
        from features.graph_utils import collate_multiscale
        graph_samples = [item["graph_sample"] for item in batch]
        result["graph_batch"] = collate_multiscale(graph_samples)

    return result


def get_scheduled_sampling_ratio(epoch: int, max_epochs: int, max_ratio: float = 0.3) -> float:
    return min(max_ratio, max_ratio * epoch / max_epochs)


def train_epoch(
    model: GeneratorDecoder,
    graph_encoder: GraphEncoder | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: GenerativeLoss,
    tokenizer: SELFIESTokenizer,
    scheduled_sampling_ratio: float,
    device: torch.device,
    prop_scaler: PropertyScaler | None = None,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    if graph_encoder is not None:
        graph_encoder.train()
    total_loss = 0.0
    n_batches = 0

    use_free_run = False
    for step, batch in enumerate(loader):
        if step == 0:
            use_free_run = random.random() < scheduled_sampling_ratio

        tokens = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)

        graph_emb = None
        if graph_encoder is not None and "graph_batch" in batch:
            graph_batch = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in batch["graph_batch"].items()
            }
            graph_emb = graph_encoder(graph_batch)

        true_property = batch.get("property")
        prop_cond = None
        if true_property is not None:
            prop_cond = true_property.to(device)

        logits, decoder_hidden = model(
            tokens, graph_emb=graph_emb, property_cond=prop_cond,
            return_hidden=True,
        )

        cls_token = decoder_hidden[:, 0, :]
        pred_property = model.property_head(cls_token).squeeze(-1)

        loss = loss_fn(
            token_logits=logits,
            targets=targets,
            pred_property=pred_property,
            true_property=prop_cond,
        )

        if use_free_run:
            prefix = tokens[:, :1]
            with torch.no_grad():
                model_out = model.generate(
                    prefix,
                    max_len=tokens.size(1),
                    graph_emb=graph_emb,
                    property_cond=prop_cond,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_tokens = model_out["tokens"]
            gen_len = min(gen_tokens.size(1), tokens.size(1))
            gen_logits, _ = model(
                gen_tokens[:, :gen_len],
                graph_emb=graph_emb,
                property_cond=prop_cond,
                return_hidden=True,
            )
            fr_loss = F.cross_entropy(
                gen_logits.reshape(-1, gen_logits.size(-1)),
                targets[:, :gen_len].reshape(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            loss = loss + 0.3 * fr_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_epoch(
    model: GeneratorDecoder,
    graph_encoder: GraphEncoder | None,
    loader: DataLoader,
    loss_fn: GenerativeLoss,
    tokenizer: SELFIESTokenizer,
    device: torch.device,
    prop_scaler: PropertyScaler | None = None,
) -> dict:
    model.eval()
    if graph_encoder is not None:
        graph_encoder.eval()
    total_loss = 0.0
    n_batches = 0
    all_pred_props = []
    all_true_props_raw = []

    for batch in loader:
        tokens = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)

        graph_emb = None
        if graph_encoder is not None and "graph_batch" in batch:
            graph_batch = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in batch["graph_batch"].items()
            }
            graph_emb = graph_encoder(graph_batch)

        true_property = batch.get("property")
        prop_cond = true_property.to(device) if true_property is not None else None

        logits, decoder_hidden = model(
            tokens, graph_emb=graph_emb, property_cond=prop_cond,
            return_hidden=True,
        )

        cls_token = decoder_hidden[:, 0, :]
        pred_property = model.property_head(cls_token).squeeze(-1)

        loss = loss_fn(
            token_logits=logits,
            targets=targets,
            pred_property=pred_property,
            true_property=prop_cond,
        )

        total_loss += loss.item()
        n_batches += 1

        if true_property is not None:
            all_pred_props.append(pred_property.cpu())
            all_true_props_raw.append(true_property.cpu())

    metrics = {"loss": total_loss / max(n_batches, 1)}
    if all_pred_props and prop_scaler is not None:
        preds_norm = torch.cat(all_pred_props).numpy()
        trues_norm = torch.cat(all_true_props_raw).numpy()
        preds = prop_scaler.inverse(preds_norm)
        trues = prop_scaler.inverse(trues_norm)
        metrics["property_rmse"] = float(np.sqrt(np.mean((preds - trues) ** 2)))
    return metrics


@torch.no_grad()
def sample_generative_metrics(
    model: GeneratorDecoder,
    graph_encoder: GraphEncoder | None,
    tokenizer: SELFIESTokenizer,
    metrics_tracker: GenerativeMetrics,
    n_samples: int,
    device: torch.device,
    decode_cfg: dict | None = None,
) -> dict:
    model.eval()
    if graph_encoder is not None:
        graph_encoder.eval()

    cfg = {**DECODE_CONFIG, **(decode_cfg or {})}
    batch_size = min(32, n_samples)
    generated = []
    n_remaining = n_samples

    while n_remaining > 0:
        bs = min(batch_size, n_remaining)
        prefix = torch.full((bs, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)

        out = model.generate(
            prefix, max_len=cfg["max_len"],
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            top_p=cfg["top_p"],
            eos_token_id=tokenizer.eos_token_id,
        )
        for tokens_i in out["tokens"].cpu().tolist():
            smi = tokenizer.decode(tokens_i)
            if smi:
                generated.append(smi)
        n_remaining -= bs

    gen_metrics = metrics_tracker.compute(generated)
    gen_metrics["n_generated"] = len(generated)
    return gen_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--model_type", default="encoder_decoder", choices=["encoder_decoder", "decoder_only"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--person", default="anon")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--auto_save_every", type=int, default=5,
                        help="Save recovery checkpoint every N epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and
                          cfg.get("device", {}).get("use_cuda", True) else "cpu")
    print(f"Device: {device}")

    ckpt_dir = Path(cfg["paths"].get("checkpoints_dir", "outputs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg["paths"]["data_dir"])

    df = pd.read_csv(data_dir / "train.csv")
    df["SMILES"] = df["SMILES"].str.strip()
    if args.max_samples:
        df = df.iloc[:args.max_samples].reset_index(drop=True)
        print(f"Limited to {len(df)} samples")

    all_smiles = df["SMILES"].tolist()
    all_properties = df["property"].tolist() if "property" in df.columns else None

    tokenizer = SELFIESTokenizer()
    tokenizer.build_vocabulary(all_smiles)
    tokenizer.save_vocab(ckpt_dir / "generator_vocab.json")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    config = GeneratorConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        use_graph_encoder=(args.model_type == "encoder_decoder"),
        graph_dim=256,
    )

    graph_encoder = None
    if config.use_graph_encoder:
        polychain_cfg = {
            "in_atom_dim": 50,
            "in_edge_dim": 8,
            "hidden_dim": 256,
            "n_backbone_layers": 4,
            "dropout": 0.1,
        }
        graph_encoder = GraphEncoder(polychain_cfg).to(device)

    model = GeneratorDecoder(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = GenerativeLoss(
        vocab_size=config.vocab_size,
        property_weight=0.5,
        label_smoothing=0.0,
        ignore_index=tokenizer.pad_token_id,
    )

    mask_fn = SELFIESMask(tokenizer)
    validator = MoleculeValidator()
    gen_metrics_tracker = GenerativeMetrics(reference_smiles=all_smiles)
    curriculum = CurriculumScheduler(df) if args.curriculum else None

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + (list(graph_encoder.parameters()) if graph_encoder else []),
        lr=args.lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_loss = float("inf")
    ckpt_tag = f"{args.person}_generator"

    if args.resume:
        best_path = ckpt_dir / f"{ckpt_tag}_best.pt"
        if best_path.exists():
            ckpt_data = load_checkpoint(best_path)
            model.load_state_dict(ckpt_data["model_state"])
            if graph_encoder is not None and "graph_state" in ckpt_data:
                graph_encoder.load_state_dict(ckpt_data["graph_state"])
            start_epoch = ckpt_data.get("epoch", 0) + 1
            best_val_loss = ckpt_data.get("val_loss", float("inf"))
            print(f"Resumed from epoch {start_epoch - 1}, best_val_loss={best_val_loss:.4f}")

    metric_tracker = MetricTracker(ckpt_dir / "generator_logs")

    n_total = len(df)
    n_val = max(1, n_total // 5)
    rng = np.random.RandomState(seed + 999)
    indices = rng.permutation(n_total)
    train_idx = set(indices[n_val:].tolist())
    val_idx = set(indices[:n_val].tolist())
    train_df = df.iloc[list(train_idx)].reset_index(drop=True)
    val_df = df.iloc[list(val_idx)].reset_index(drop=True)
    train_smiles = train_df["SMILES"].tolist()
    val_smiles = val_df["SMILES"].tolist()
    train_props = train_df["property"].tolist() if all_properties else None
    val_props = val_df["property"].tolist() if all_properties else None

    prop_scaler = PropertyScaler()
    if train_props is not None:
        prop_scaler.fit(train_props)
        train_props_processed = prop_scaler.transform(train_props).tolist()
        val_props_processed = prop_scaler.transform(val_props).tolist() if val_props else None
        print(f"Property scaler: mean={prop_scaler.mean:.4f}, std={prop_scaler.std:.4f}")
    else:
        train_props_processed = None
        val_props_processed = None

    for phase in range(6):
        if args.curriculum:
            phase_df = curriculum.get_subset(phase)
            phase_smiles_set = set(phase_df["SMILES"].str.strip().tolist())
            phase_train = [s for s in train_smiles if s in phase_smiles_set]
            phase_val = [s for s in val_smiles if s in phase_smiles_set]
            phase_train_props = (
                [train_props_processed[train_smiles.index(s)] for s in phase_train]
                if train_props_processed else None
            )
            phase_val_props = (
                [val_props_processed[val_smiles.index(s)] for s in phase_val]
                if val_props_processed else None
            )
            if not phase_train:
                phase_train = train_smiles
                phase_val = val_smiles
                phase_train_props = train_props_processed
                phase_val_props = val_props_processed
            print(f"Curriculum phase {phase}: {len(phase_train)} train, {len(phase_val)} val")
        else:
            phase_train = train_smiles
            phase_val = val_smiles
            phase_train_props = train_props_processed
            phase_val_props = val_props_processed

        train_dataset = SMILESDataset(
            phase_train, tokenizer, phase_train_props,
            build_graphs=(graph_encoder is not None),
        )
        val_dataset = SMILESDataset(
            phase_val, tokenizer, phase_val_props,
            build_graphs=(graph_encoder is not None),
        )

        if len(train_dataset) == 0:
            print(f"  Skipping phase {phase}: no valid training samples after filtering")
            continue

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_generator,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_generator,
        )

        for epoch in range(start_epoch, args.epochs + 1):
            ss_ratio = get_scheduled_sampling_ratio(epoch, args.epochs)
            train_loss = train_epoch(
                model, graph_encoder, train_loader, optimizer, loss_fn,
                tokenizer, ss_ratio, device,
            )
            val_metrics = validate_epoch(
                model, graph_encoder, val_loader, loss_fn, tokenizer, device,
                prop_scaler=prop_scaler,
            )
            scheduler.step()

            val_loss = val_metrics["loss"]
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            perplexity = math.exp(min(val_loss, 20))

            metrics = {
                "phase": phase,
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "perplexity": round(perplexity, 2),
                "lr": round(scheduler.get_last_lr()[0], 8),
            }
            if "property_rmse" in val_metrics:
                metrics["property_rmse"] = round(val_metrics["property_rmse"], 4)

            if epoch % 10 == 0 or epoch == 1:
                gen_metrics = sample_generative_metrics(
                    model, graph_encoder, tokenizer, gen_metrics_tracker,
                    n_samples=50, device=device,
                )
                metrics["validity"] = gen_metrics["validity"]
                metrics["uniqueness"] = gen_metrics["uniqueness"]
                metrics["novelty"] = gen_metrics["novelty"]
                metrics["scaffold_diversity"] = gen_metrics["scaffold_diversity"]

                print(f"Phase {phase} | Epoch {epoch:3d}/{args.epochs} | "
                      f"loss={train_loss:.4f}/{val_loss:.4f} | ppl={perplexity:.2f} | "
                      f"valid={gen_metrics['validity']:.2f} uniq={gen_metrics['uniqueness']:.2f} "
                      f"novel={gen_metrics['novelty']:.2f} div={gen_metrics['scaffold_diversity']:.2f}" +
                      (f" | prop_rmse={metrics['property_rmse']}" if "property_rmse" in metrics else ""))
            else:
                print(f"Phase {phase} | Epoch {epoch:3d}/{args.epochs} | "
                      f"loss={train_loss:.4f}/{val_loss:.4f} | ppl={perplexity:.2f}")

            metric_tracker.log(epoch, metrics)

            if is_best:
                ckpt_payload = {
                    "model_state": model.state_dict(),
                    "graph_state": graph_encoder.state_dict() if graph_encoder else None,
                    "config": {
                        "vocab_size": config.vocab_size,
                        "d_model": config.d_model,
                        "n_head": config.n_head,
                        "n_layer": config.n_layer,
                        "use_graph_encoder": config.use_graph_encoder,
                        "graph_dim": config.graph_dim,
                    },
                    "prop_scaler": prop_scaler.state_dict() if prop_scaler.fitted else None,
                    "epoch": epoch,
                    "phase": phase,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "vocab": tokenizer._vocab,
                }
                save_checkpoint(ckpt_payload, ckpt_dir / f"{ckpt_tag}_phase{phase}_best.pt")

            if epoch % args.auto_save_every == 0:
                recovery_payload = {
                    "model_state": model.state_dict(),
                    "graph_state": graph_encoder.state_dict() if graph_encoder else None,
                    "config": {
                        "vocab_size": config.vocab_size,
                        "d_model": config.d_model,
                        "n_head": config.n_head,
                        "n_layer": config.n_layer,
                        "use_graph_encoder": config.use_graph_encoder,
                        "graph_dim": config.graph_dim,
                    },
                    "prop_scaler": prop_scaler.state_dict() if prop_scaler.fitted else None,
                    "epoch": epoch,
                    "phase": phase,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "vocab": tokenizer._vocab,
                }
                save_checkpoint(recovery_payload, ckpt_dir / f"{ckpt_tag}_recovery.pt")
                print(f"  Auto-saved recovery checkpoint at epoch {epoch}")

        start_epoch = 1

    final_state = {
        "model_state": model.state_dict(),
        "graph_state": graph_encoder.state_dict() if graph_encoder else None,
        "config": {
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "use_graph_encoder": config.use_graph_encoder,
            "graph_dim": config.graph_dim,
        },
        "prop_scaler": prop_scaler.state_dict() if prop_scaler.fitted else None,
        "epoch": args.epochs,
        "val_loss": best_val_loss,
        "vocab": tokenizer._vocab,
    }
    save_checkpoint(final_state, ckpt_dir / f"{ckpt_tag}_final.pt")
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")

    try:
        from notebooks.generative_viz import generate_all_plots
        generate_all_plots(
            data_path=str(data_dir / "train.csv"),
            log_path=str(ckpt_dir / "generator_logs"),
            save_dir=str(Path(cfg["paths"].get("reports_dir", "reports")) / "figures" / "generator"),
            quiet=True,
        )
    except Exception as e:
        print(f"Visualization generation skipped ({e})")


if __name__ == "__main__":
    main()
