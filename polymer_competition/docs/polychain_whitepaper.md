# PolyChain: Hierarchical Periodic Transformer with Equivariant Multi-Scale Graph Reasoning for Polymer Property Prediction

> **Full Research Proposal — ML4Materials Workshop**

---

## Abstract

We introduce **PolyChain**, a novel deep-learning architecture that predicts polymer properties directly from monomer SMILES by jointly modeling three scales of polymer structure that prior work has treated in isolation: (i) the *monomer* graph, (ii) the *periodic* extension of the chain, and (iii) the *chain-statistical* context (sequence length, branching, end-groups).

PolyChain's two central innovations are:
1. **Periodic Equivariant Chain-Growth Network (PECGN)** — treats a polymer as an unbounded periodic graph and uses a learnable boundary operator to make predictions invariant to where the SMILES is "cut."
2. **Hierarchy-Aware Multi-Scale Fusion (HAMF)** — propagates information between monomer, dimer, and trimer views through a chain-structured cross-attention mechanism, then injects a polymer-specific **Chain Statistics Token (CST)** derived from sequence, branching, and end-group features.

We further propose a **self-supervised periodicity pretraining** objective — *asterisk-mask reconstruction* — that forces the model to recover the periodic structure of the chain from masked connection points.

---

## 1. Architecture

### 1.1 Multi-Scale Graph Construction

From a polymer SMILES like `*CCO*`, we construct four graph views:

| Scale | Description | Implementation |
|-------|-------------|----------------|
| Monomer (k=1) | Literal atom-bond graph | `features/graphs.py::smiles_to_graph()` |
| Dimer (k=2) | Two concatenated repeats | `features/graphs.py::kmer_graph(k=2)` |
| Trimer (k=3) | Three concatenated repeats | `features/graphs.py::kmer_graph(k=3)` |
| Periodic (k=∞) | Closed ring (Antoniuk-style) | `features/graphs.py::periodic_graph()` |

### 1.2 GIN-S Backbone

An **Edge-aware Graph Isomorphism Network with virtual supernode** (GIN-S) serves as the shared encoder for all scales.

- **Virtual node**: A learnable parameter that mediates global information exchange across the graph, updated at each message-passing layer.
- **Edge features**: Bond type (one-hot), conjugation, ring membership, and a boundary flag for `*`-crossing bonds.
- **Output**: Graph-level embedding `h_k ∈ ℝ^d` via `global_add_pool`.

Implementation: `models/polychain/backbone.py`

### 1.3 HAMF — Hierarchy-Aware Multi-Scale Fusion

The first core innovation. Treats the three scale embeddings `[h₁, h₂, h₃]` as a sequence and applies a **chain-structured transformer** with causal masking (scale k attends to scales ≤ k):

```
h̃_k = LayerNorm(h_k + MHA_k({h_j}_{j≤k}))
H_MS = [h̃₁ ⊕ h̃₂ ⊕ h̃₃] ∈ ℝ^{3d}
```

Key design choices:
- **Causal mask**: Reflects that longer oligomers subsume shorter ones
- **Learnable scale positional encoding**: Added before attention
- **Pre-norm architecture**: LayerNorm before attention and FFN blocks

Implementation: `models/polychain/hamf.py`

### 1.4 CST — Chain Statistics Token

A fixed-dimensional vector computed from SMILES alone:
- Effective repeat length, branching indicator
- End-group statistics (OH, COOH, NH₂, halides, vinyl, etc.)
- Ring statistics (count, sizes, aromaticity)
- Backbone heteroatom flag, molecular weight
- Copolymer monomer count

Normalized via calibration-set z-scoring and projected to hidden dim.

Implementation: `models/polychain/cst.py`

### 1.5 PECGN — Periodic Equivariant Chain-Growth Network

The second core innovation. Replaces Antoniuk's hard-wired periodic bond with a **learned, direction-aware boundary operator**:

```
h_periodic = h_trimer + α · BoundaryOp(h_trimer, direction, c)
```

**Invariance properties** (by construction):
- **Translation invariance**: BoundaryOp applied symmetrically at both ends → invariant to shifting the SMILES cut point
- **Permutation invariance within repeats**: Inherited from GIN aggregator
- **Sequence-length robustness**: α is clamped to small values (≤ 0.3)

Implementation: `models/polychain/pecgn.py`

### 1.6 End-to-End Model

```
forward(batch_dict):
    h1 = backbone(monomer)    # (B, d)
    h2 = backbone(dimer)      # (B, d)
    h3 = backbone(trimer)     # (B, d)
    fused = HAMF([h1, h2, h3])  # (B, 3d)
    cst_emb = CST_norm(cst)     # (B, d)
    periodic = PECGN(fused, cst_emb)  # (B, 3d)
    ŷ = MLP([periodic ⊕ cst_emb])    # (B,)
```

Implementation: `models/polychain/polychain_model.py`

---

## 2. Self-Supervised Pretraining

### 2.1 Asterisk-Mask Reconstruction (Primary)

Randomly mask 0–2 `*` connection points per chain. The model must predict:
1. The **type** of each masked `*` (left-end / right-end / internal)
2. The **identity** of its neighbor atom

Loss: cross-entropy on `*` type + cross-entropy on neighbor element.

This is a polymer-specific pretext task with no analog in molecular SSL.

Implementation: `models/polychain/pretraining/asterisk_mask.py`

### 2.2 Sub-SMILES Masking (Auxiliary)

Standard BERT-style atom masking on SMILES tokens: 15% of non-special tokens are masked (80% [MASK], 10% random, 10% unchanged).

Implementation: `models/polychain/pretraining/sub_smiles_mask.py`

---

## 3. Training Strategy

### 3.1 Fine-Tuning

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine annealing with 5-epoch warmup
- **Early stopping**: Patience 30 on validation RMSE
- **Gradient clipping**: Max norm 1.0
- **Batch size**: 32

### 3.2 Loss Function

```
L = L_MSE(y, ŷ) + λ₁·L_aux_pretrain + λ₂·L_CST_pred
```

### 3.3 Cross-Validation

Group K-Fold by SMILES scaffold (5 folds), using `features/build_features.py::make_splits()`.

### 3.4 Data Augmentation

- SMILES randomization (5× augmentation)
- Chain-extension augmentation
- Substructure mixing (copolymers)

---

## 4. Expected Results

| Property | Uni-Poly R² | PolyChain Expected R² | Expected Δ RMSE |
|----------|-------------|----------------------|-----------------|
| **Tg** | 0.921 | 0.93–0.95 | -5 to -10% |
| **Density** | 0.823 | 0.86–0.89 | -7 to -10% |
| **Td** | 0.775 | 0.82–0.85 | -5 to -8% |
| **Tm** | 0.618 | 0.70–0.76 | **-10 to -18%** |
| **Er** | 0.588 | 0.64–0.70 | -7 to -12% |

---

## 5. Ablation Plan

1. **w/o PECGN** — replace with Antoniuk's fixed periodic graph
2. **w/o HAMF** — replace with simple concatenation
3. **w/o CST** — remove chain statistics
4. **w/o pretraining** — skip asterisk-mask SSL
5. **Single scale only** (k=1, k=2, or k=3)
6. **w/o SMILES augmentation**

---

## 6. Limitations

- No 3D / morphology modeling
- No copolymer sequence modeling (set-pooling only)
- Single property per training run
- Process conditions ignored
- Long-range interactions (>15 repeats) not modeled

---

## References

1. Queen et al. (2023) — POLYMERGNN
2. Park et al. (2022) — GCN with Attention
3. Antoniuk et al. (2022) — Periodic Polymer Graph
4. Huang et al. (2025) — Uni-Poly
5. Xu et al. (2018) — GIN
6. Kipf & Welling (2017) — GCN
