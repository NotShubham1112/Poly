# PolyChain – Architecture Overview

## High-level system design

```
        ┌──────────────────────────────────────────────────┐
        │                  Raw Data                        │
        │      train.csv / test.csv (SMILES + y)           │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │     features/build_features.py                   │
        │  ─ fingerprints (Morgan, MACCS, atom-pair)       │
        │  ─ RDKit descriptors (~200)                      │
        │  ─ polymer-specific features (CST base)          │
        │  ─ imputation + Group K-Fold splits              │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │   models/         (per-model architectures)      │
        │  ─ baselines, tree, MLP, GNN, GraphTransformer   │
        │  ─ fusionnet  (multimodal cross-attention)       │
        │  ─ polychain  (★ hierarchical periodic)          │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │   training/train.py                              │
        │  per-fold training, OOF predictions → .pkl       │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │   ensemble/build_ensemble.py                     │
        │  weight optimization (inverse-rme / NM / stack)  │
        │  → outputs/submissions/submission.csv            │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │   inference/predictor.py + chat_interface.py     │
        │  trained model deployed for ad-hoc predictions   │
        └──────────────────────────────────────────────────┘
```

## PolyChain sub-modules (★)

```
SMILES ──► graph_builder ──► backbone (GIN-S, shared)
                                  │
                                  ▼
                       scale embeddings (mono/di/tri)
                                  │
                                  ▼
                     hamf (cross-scale attention)
                                  │
                                  ▼
                        fused multi-scale repr
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
      pecgn (boundary operator)            cst (chain stats)
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                          prediction head
```

See `polychain_whitepaper.md` for the full mathematical description.
