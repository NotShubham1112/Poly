# PolyChain: Hierarchical Periodic Transformer with Equivariant Multi-Scale Graph Reasoning for Polymer Property Prediction

> **A novel deep-learning architecture that predicts polymer properties directly from monomer SMILES by jointly modeling monomer, dimer, trimer, and periodic chain scales — without external LLM augmentation.**

![Status](https://img.shields.io/badge/status-research--proposal-blue) ![Domain](https://img.shields.io/badge/domain-polymer%20informatics-green) ![Venue](https://img.shields.io/badge/target-ML4Materials%20workshop-orange)

---

## Table of Contents

 1. [Abstract](#abstract)
 2. [Motivation & Problem Statement](#1-motivation--problem-statement)
 3. [Related Work (Critical Analysis)](#2-related-work-critical-analysis)
 4. [Proposed Architecture](#3-proposed-architecture)
 5. [Training Strategy & Implementation Plan](#4-training-strategy--implementation-plan)
 6. [Evaluation & Expected Results](#5-evaluation--expected-results)
 7. [Limitations & Future Work](#6-limitations--future-work)
 8. [Conclusion](#7-conclusion)
 9. [References](#8-references)
10. [Repository Structure](#repository-structure)
11. [Quick Start](#quick-start)
12. [Citation](#citation)

---

## Abstract

We introduce **PolyChain**, a novel deep-learning architecture that predicts polymer properties directly from monomer SMILES by jointly modeling three scales of polymer structure that prior work has treated in isolation: (i) the *monomer* graph, (ii) the *periodic* extension of the chain, and (iii) the *chain-statistical* context (sequence length, branching, end-groups). PolyChain's two central innovations are **(1) a Periodic Equivariant Chain-Growth Network (PECGN)** that treats a polymer as an unbounded periodic graph and uses a learnable boundary operator to make predictions invariant to where the SMILES is "cut," and **(2) a Hierarchy-Aware Multi-Scale Fusion (HAMF) module** that propagates information between monomer, dimer, and trimer views through a chain-structured cross-attention mechanism, then injects a polymer-specific **Chain Statistics Token (CST)** derived from sequence, branching, and end-group features. We further propose a **self-supervised periodicity pretraining** objective—*asterisk-mask reconstruction*—that forces the model to recover the periodic structure of the chain from masked connection points. The architecture is designed to be trained from SMILES alone, making it directly applicable to competition data. We hypothesize a 5–10% relative RMSE reduction on Tg, 10–18% on Tm, and 7–12% on density versus Uni-Poly, with the largest gains on properties sensitive to chain length and crystallinity.

---

## 1. Motivation & Problem Statement

Polymer materials underpin virtually every sustainable technology of the coming decade: recyclable packaging, biocompatible implants, lightweight vehicles, and solid-state batteries. Yet experimental screening of polymer candidates remains a 6–18 month endeavor. **ML-driven property prediction from SMILES is the most scalable surrogate**, but four persistent gaps—visible across the four reference papers—block its reliability:

### 1.1 Monomer-only view

**POLYMERGNN** (Queen et al., 2023) and **Park et al.** (2022) both treat polymers as bags of isolated monomers. Park et al.'s GCN-Attention model, while interpretable, never sees two repeat units simultaneously. This ignores that a polymer's *identity* is partly defined by the bonds it forms with itself—a fact **Uni-Poly** (Huang et al., 2025) explicitly laments.

### 1.2 Inconsistent periodic featurization

**Antoniuk et al.** (2022) demonstrate that `*CCO*` and `*COC*` produce different RDKit feature vectors despite representing the same polymer. Their periodic graph fix is correct but is restricted to a *single* chain length and treats the boundary as if it did not exist.

### 1.3 No multi-scale fusion

**Uni-Poly** combines 5 modalities (SMILES, 2D, 3D, FP, text) but all operate at the *monomer* scale. It achieves R² ≈ 0.62 on Tm, R² ≈ 0.59 on Er—the *lowest* scores—precisely because chain-scale phenomena (crystallinity, entanglements, packing) are invisible to monomer-only features.

### 1.4 Hand-crafted or LLM-dependent text augmentation

Uni-Poly relies on GPT-4-generated captions that are inherently noisy, with high cross-run variance, and unavailable for novel test-time SMILES.

### The Gap PolyChain Fills

**PolyChain** is a single architecture that:

- ✅ Is periodicity-equivariant by construction
- ✅ Explicitly fuses monomer → dimer → trimer scales
- ✅ Injects chain statistics as a learnable token
- ✅ Trains end-to-end from SMILES without external caption generation

---

## 2. Related Work (Critical Analysis)

### 2.1 Park et al. (2022) — GCN with Attention

| Aspect | Detail |
| --- | --- |
| **Contribution** | A GCN operating on the monomer graph, augmented with an attention pooling layer for interpretability, applied to Tg, density, and Tm on PoLyInfo. |
| **R² (Tg)** | &gt; 0.85 |

**Weaknesses overcome by PolyChain:**

- Predicts from a *single* monomer graph with no chain context.
- Uses canonical SMILES only, inheriting Antoniuk's periodicity inconsistency.
- Attention is used solely for explanation, not for fusing structural information.

**PolyChain difference:** We use attention to *fuse* monomer, dimer, trimer embeddings—not merely to visualize one.

### 2.2 Antoniuk et al. (2022) — Periodic Polymer Graph

| Aspect | Detail |
| --- | --- |
| **Contribution** | Modifies Chemprop's featurizer to wrap a *k*-repeat chain and connect terminal atoms, restoring periodicity invariance. Periodic graphs outperform monomer, dimer, and trimer graphs. |

**Weaknesses overcome by PolyChain:**

- Periodic wrapping is a *graph construction* trick; the MPNN is unchanged and remains permutation-invariant to chain orientation, discarding directional chain information.
- They use a *fixed* repeat count (k = 1 or 3); no mechanism to *learn* the optimal scale.
- Modest improvement on Tm and density.

**PolyChain difference:** PECGN's boundary operator is *learned* and *direction-aware*, and HAMF adaptively weights monomer/dimer/trimer rather than picking one.

### 2.3 Queen et al. (2023) — POLYMERGNN

| Aspect | Detail |
| --- | --- |
| **Contribution** | A multitask GNN with separate acid/glycol embedding blocks, self-attention pooling, and an explicit log–log prediction head for Tg. R² (Tg) = 0.86. |

**Weaknesses overcome by PolyChain:**

- Treats each monomer in isolation; never sees the ester bond between acid and glycol.
- Adds resin properties (AN, OHN, Mw) as static scalars—these are *not* available for unseen SMILES at test time.
- No pretraining strategy.

**PolyChain difference:** PolyChain's CST is computed *from SMILES alone* (e.g., via functional group counts) and is therefore usable at inference, while still being polymer-aware.

### 2.4 Huang et al. (2025) — Uni-Poly

| Aspect | Detail |
| --- | --- |
| **Contribution** | Multimodal fusion of SMILES (ChemBERTa), 2D (GIN), 3D (SchNet), Morgan FP, and LLM-generated captions, with cross-modal contrastive pretraining. R² (Tg) ≈ 0.92. |

**Weaknesses overcome by PolyChain:**

- All modalities operate at the *monomer* scale; the authors explicitly identify this as the bottleneck for Tm and Er.
- Caption generation requires an external LLM call per polymer, with reproducibility and cost issues.
- Cannot handle copolymers (acknowledged limitation).
- No chain-equivariance; periodicity inconsistency from RDKit featurization persists.

**PolyChain difference:** PolyChain is *unimodal at input* (SMILES only) but *multiscale internally*, sidestepping LLM dependency and the modality-alignment problem while attacking the bottleneck Uni-Poly identifies.

### 2.5 Comparative Summary

| Method | Representation | Polymer-specific invariance | Multi-scale | Interpretable | Data-efficient | SMILES-only at test time |
| --- | --- | --- | --- | --- | --- | --- |
| Park 2022 (GCN-Att) | 2D graph (monomer) | None | ✗ | ✓ | ✗ | ✓ |
| Antoniuk 2022 (PPG) | Periodic graph (k=1,3) | Translation along backbone | ✗ | ✗ | ✗ | ✓ |
| POLYMERGNN 2023 | Set of monomer graphs | Set permutation | ✗ | ✓ | ✓ | ✗ |
| Uni-Poly 2025 | 5 modalities × monomer | None | ✗ | Partial | ✓ | ✗ |
| **PolyChain (ours)** | SMILES → monomer/dimer/trimer/periodic graph + CST | **Translation + ASTERISK-shift** | **✓** | **✓** | **✓** | **✓** |

---

## 3. Proposed Architecture

### 3.1 High-Level Design

PolyChain is composed of five sequential modules. The data flow is illustrated below.

```
                ┌─────────────────────────────────────────────────────┐
                │   SMILES  ──►  Tokenizer  ──►  Atom/Bond Features  │
                └──────────────────────┬──────────────────────────────┘
                                       │
        ┌──────────────────────────────┼───────────────────────────────┐
        ▼                              ▼                               ▼
 ┌──────────────┐              ┌──────────────┐                 ┌──────────────┐
 │ Monomer GNN  │              │ Dimer GNN    │                 │ Trimer GNN   │
 │    (k=1)     │              │    (k=2)     │                 │    (k=3)     │
 └──────┬───────┘              └──────┬───────┘                 └──────┬───────┘
        │                             │                               │
        └──────────────┬──────────────┴───────────────┬───────────────┘
                       ▼                              ▼
            ┌──────────────────────┐        ┌──────────────────────┐
            │   HAMF (cross-attn)  │        │   CST (chain stats)  │
            │   fuses 3 scales     │        │   - length, branch,  │
            │                      │        │     end-groups, rings│
            └──────────┬───────────┘        └──────────┬───────────┘
                       │                               │
                       └───────────────┬───────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │  Periodic Equivariant Output  │
                       │      (PECGN) + Task Head      │
                       └───────────────────────────────┘
```

### 3.2 Component 1 — SMILES Tokenizer and Graph Constructor

We tokenize SMILES into a sequence of atoms and connection symbols. From a SMILES like `*CCO*` (polyethylene glycol repeat unit), we construct **four graph views**:

- **Monomer graph (k=1):** the literal atom-bond graph of the SMILES, with `*` atoms marked but not featurized as chemical elements.
- **Dimer graph (k=2):** duplicate the repeat unit, connect the right `*` of the first copy to the left `*` of the second copy.
- **Trimer graph (k=3):** analogous.
- **Periodic graph (k=∞ proxy):** connect *both* `*` atoms, treating the chain as a closed ring (Antoniuk's PPG) but with a *learned* boundary.

**Atom features:** atomic number, degree, formal charge, hybridization, aromaticity, in-ring flag, and a learnable embedding for `*` (with separate embeddings for left and right connection points).

**Bond features:** bond type, in-ring, and a flag indicating whether the bond crosses a `*` boundary.

### 3.3 Component 2 — Multi-Scale GNN Backbone

Each of the three graph views is processed by an **Edge-aware Graph Isomorphism Network with subgraph counting (GIN-S)**, with a virtual node carrying the global graph state. For view *k* ∈ {1, 2, 3}, the readout yields a graph-level embedding **h_k** ∈ ℝ^d.

The choice of GIN is deliberate: it is the most expressive message-passing GNN under the Weisfeiler–Leman test, and its injection of neighbor features through sum aggregation complements our later cross-scale fusion (which provides the order-awareness GIN itself lacks).

**Message-passing equations** (for each scale k):

$$ \\mathbf{m}\_{ij}^{(l)} = \\phi_e!\\left(\\mathbf{h}\_i^{(l)},\\mathbf{h}*j^{(l)},\\mathbf{e}*{ij}\\right) $$

$$ \\mathbf{h}*i^{(l+1)} = \\phi_h!\\left(\\mathbf{h}i^{(l)},;\\sum{j\\in\\mathcal{N}(i)}\\mathbf{m}*{ij}^{(l)}\\right) $$

where ϕ\_e and ϕ\_h are 2-layer MLPs and e_ij carries the `*`-boundary flag. After L layers, a virtual node aggregation yields the scale embedding h_k.

### 3.4 Innovation #1 — Hierarchy-Aware Multi-Scale Fusion (HAMF)

This is the **first fundamentally new component**. Standard approaches either pick a single scale (Antoniuk 2022) or simply concatenate scale embeddings. HAMF treats the three scale embeddings as a *sequence* and applies a chain-structured transformer over them:

1. **Scale positional encoding** s_k is added to each h_k.
2. A bidirectional cross-attention block with a causal-like mask: scale k can attend to scales ≤ k, reflecting that longer oligomers subsume shorter ones.
3. The output is the concatenation \[h̃\_1 ; h̃\_2 ; h̃\_3\], which captures both local motifs and their context.

**Cross-attention equation:**

$$ \\tilde{\\mathbf{h}}\_k = \\text{LayerNorm}!\\left(\\mathbf{h}\_k + \\text{MHA}\_k!\\left({\\mathbf{h}*j}*{j\\le k}\\right)\\right) $$

with multi-head attention over the *scale* axis (not the atom axis). The final multi-scale embedding is:

$$ \\mathbf{H}\_{\\text{MS}} = \[\\tilde{\\mathbf{h}}\_1 \\oplus \\tilde{\\mathbf{h}}\_2 \\oplus \\tilde{\\mathbf{h}}\_3\] \\in \\mathbb{R}^{3d} $$

**Why new:** No prior work has applied cross-attention across *oligomer scales* as if they were tokens in a sequence. Uni-Poly applies cross-attention across *modalities*; Antoniuk picks a single scale; POLYMERGNN sums monomer embeddings. HAMF is the first to model the *compositional* relationship between scales.

### 3.5 Innovation #2 — Chain Statistics Token (CST) and the Periodic Equivariant Chain-Growth Network (PECGN)

This is the **second fundamentally new component**.

#### CST computation

From SMILES alone (no external property data), we compute:

- **Effective repeat length** ℓ = number of non-`*` heavy atoms in the SMILES.
- **Branching indicator** b = 1 if any atom has degree &gt; 2 in the monomer, else 0; for copolymers, a distribution of b over monomers.
- **End-group statistics:** counts of OH, COOH, NH₂, halide, vinyl groups (computed by SMARTS substructure matching on a *synthesized* chain of k = 3 repeats).
- **Ring statistics:** number and sizes of rings; fraction of aromatic carbons.

These are normalized and embedded into a fixed-dim vector **c** ∈ ℝ^d.

#### PECGN (Periodic Equivariant Chain-Growth Network)

PECGN replaces Antoniuk's hard-wired boundary bond with a *learned, direction-aware* boundary operator. For an oligomer with k repeats, PECGN constructs the final embedding:

$$ \\mathbf{h}*{\\text{periodic}} = \\mathbf{h}*{\\text{trimer}} + \\alpha \\cdot \\text{BoundaryOp}(\\mathbf{h}\_{\\text{trimer}},; \\text{direction},; \\mathbf{c}) $$

where **direction** ∈ {left, right} is a learnable flag indicating which terminus is "extended," and α is a learned scalar gate. **BoundaryOp** is a 2-layer MLP that injects a relative-position signal encoding "this is the kth repeat from the chain end."

#### Invariance properties (proven by construction)

- **Translation invariance:** Because BoundaryOp is applied symmetrically at both ends, h_periodic is invariant to shifting the SMILES cut point by one repeat (e.g., `*CCO*` ⇔ `*COC*`).
- **Permutation invariance within repeat units:** Inherited from the GIN aggregator.
- **Sequence-length robustness:** PECGN uses h_trimer as a *proxy* for the infinite chain; the boundary gate α is clamped to small values during training, so the embedding is dominated by the bulk.

#### Prediction head

$$ \\hat{y} = \\text{MLP}!\\left(\[\\mathbf{H}\_{\\text{MS}} \\oplus \\mathbf{c}\]\\right) \\in \\mathbb{R} $$

### 3.6 Handling of Variable Monomers, Branching, and OOD

- **Copolymers:** CST's branching indicator naturally generalizes; HAMF can be re-run with one scale per monomer and combined via a permutation-invariant pooling over monomers (mirroring POLYMERGNN's acid/glycol split, but applied to *any* monomer set).
- **Branching:** Explicitly encoded in CST and in the monomer graph's degree features.
- **Variable monomer count:** Handled by set-pooling (sum or attention) over per-monomer embeddings, in the same spirit as POLYMERGNN, but applied after HAMF (so each monomer's multi-scale context is computed independently).
- **OOD robustness:** The periodicity pretraining (next subsection) forces the model to reconstruct masked `*` atoms from context, regularizing it to learn periodicity-invariance rather than memorize specific repeat units. This should improve transfer to monomers with backbones unseen in training.

---

## 4. Training Strategy & Implementation Plan

### 4.1 Week-by-Week Timeline

#### **Week 1 — Data and Graph Construction**

- Parse all train+test SMILES from the competition dataset and PolyInfo.
- Implement monomer/dimer/trimer/periodic graph constructors in a single Python module.
- Verify chemistry: spot-check 50 SMILES for correct atom counts, no `*` atoms treated as elements, correct ring detection.
- Build a baseline GIN-on-monomer (no PolyChain additions) to confirm the pipeline runs.

#### **Week 2 — Self-Supervised Pretraining + Fine-Tuning**

- **Asterisk-mask reconstruction pretraining.** Randomly mask 0/1/2/3 `*` atoms in each chain; train PECGN to predict the *type* of the masked `*` and the *identity* of its neighbor atom. Loss: cross-entropy on `*` type + cross-entropy on neighbor element. This is a polymer-specific pretext task with no analog in molecular SSL.
- **Masked sub-SMILES pretraining (auxiliary).** Standard BERT-style atom masking on the SMILES tokens.
- Train both objectives jointly for \~50 epochs on *all* SMILES (train + test, labels not used).
- **Fine-tune** on the labeled training set with property prediction loss for 200 epochs, early-stopping on validation R².

#### **Week 3 — Hyperparameter Tuning and Ensembling**

- Sweep: d ∈ {128, 256}, L ∈ {3, 4, 5}, k ∈ {1, 2, 3} (use subset of scales), α gate initialization, CST dimension, pretrain epochs.
- 5-fold cross-validation on the training set with grouped splits (by SMILES cluster) to avoid leakage.
- Train 5 seeds of the best configuration; average predictions.
- Explore stacking with Uni-Poly and POLYMERGNN baselines (concatenation of OOF predictions → ridge regression).

#### **Week 4 — Final Training, Submission, Paper**

- Retrain on the full training set with the best hyperparameters.
- Generate test predictions and a confidence interval from the ensemble.
- Write up ablations, the architecture diagram, and a 4-page workshop paper.

### 4.2 Hardware and Runtime

| Resource | Specification | Estimated Time |
| --- | --- | --- |
| **GPU** | One 16 GB GPU (RTX 4080 / A4000 / Kaggle P100) | — |
| **Pretraining** | — | \~6 hours |
| **Fine-tuning per fold** | — | \~40 minutes |
| **Full ensemble (5 seeds × 5 folds)** | — | \~12 hours |
| **RAM** | 32 GB (holds \~5 GB compressed PolyInfo graph cache) | — |
| **Storage** | \~20 GB for pretraining checkpoints | — |

### 4.3 Data Augmentation

- **SMILES randomization** (multiple equivalent SMILES per polymer, used in training only) — 5× augmentation.
- **Repeat-unit augmentation:** For the same monomer, alternate k=1, k=2, k=3 views in training so HAMF sees consistent targets across scales.
- **Chain-extension augmentation:** Pad the SMILES with one extra repeat on each side periodically; this trains PECGN's boundary operator to handle variable chain length.
- **Substructure mixing augmentation** (copolymer support): For co-polymers, randomly shuffle the order of monomers in the input set and verify the output is invariant.

### 4.4 Loss Function

Total loss for fine-tuning:

$$ \\mathcal{L} = \\mathcal{L}*{\\text{MSE}}(y, \\hat{y}) + \\lambda_1 \\mathcal{L}*{\\text{aux-pretrain}} + \\lambda_2 \\mathcal{L}\_{\\text{CST-pred}} $$

where $\\mathcal{L}\_{\\text{CST-pred}}$ is a multi-task auxiliary head that predicts ring count, end-group counts, and backbone aromaticity from the multi-scale embedding (these are computable from SMILES and serve as a regularizer). For multi-property datasets, we use a multi-task MSE sum.

### 4.5 Cross-Validation Strategy

- **Group K-Fold** by SMILES scaffold (RDKit Murcko scaffold on the monomer), 5 folds. This prevents scaffold leakage and gives realistic OOD estimates.
- A second split by *property value quantile* to detect distribution shift.

---

## 5. Evaluation & Expected Results

### 5.1 Metrics

- **Primary:** RMSE and MAE (competition default).
- **Secondary:** R², Spearman ρ, and a **calibration metric** (predicted vs. observed decile plot) to flag OOD failures.
- **Robustness:** Performance on a held-out "high-Mw" and "highly aromatic" subset, where chain effects are strongest.

### 5.2 Baselines

We compare PolyChain against:

- **POLYMERGNN** (Queen et al., 2023) — re-implemented from scratch.
- **Uni-Poly** (Huang et al., 2025) — using released code if available; else re-implement the encoder/contrastive pipeline.
- **Antoniuk 2022 periodic graph MPNN** (the strongest unimodal baseline).
- **A handcrafted fingerprint MLP** (Morgan + RDKit descriptors) — the "classical" baseline.

All baselines are retrained on the *same* training split with the *same* seeds to ensure fair comparison.

### 5.3 Expected Improvements

Based on the pattern of Antoniuk 2022 (where periodic &gt; trimer &gt; monomer on most properties) and Uni-Poly 2025 (where multimodal &gt; unimodal by 1.1–5.1% on R²), we hypothesize:

| Property | Uni-Poly R² (reported) | PolyChain expected R² | Expected Δ RMSE |
| --- | --- | --- | --- |
| **Tg** | 0.921 | 0.93 – 0.95 | \-5 to -10% |
| **Density** | 0.823 | 0.86 – 0.89 | \-7 to -10% |
| **Td** | 0.775 | 0.82 – 0.85 | \-5 to -8% |
| **Tm** | 0.618 | 0.70 – 0.76 | **-10 to -18%** |
| **Er** | 0.588 | 0.64 – 0.70 | \-7 to -12% |

The largest gains are on **Tm** (chain crystallinity is a chain-scale phenomenon, invisible to monomer-only models) and **Er** (electronic conduction depends on conjugation length, which dimer/trimer views capture).

### 5.4 Ablation Studies

We will report the following ablations on a 5-fold CV:

1. **w/o PECGN** (replace with Antoniuk's fixed periodic graph) — quantifies contribution of the *learned* boundary.
2. **w/o HAMF** (replace with simple concatenation) — quantifies contribution of cross-scale attention.
3. **w/o CST** — quantifies contribution of chain statistics.
4. **w/o pretraining** — quantifies contribution of asterisk-mask SSL.
5. **Single scale only** (k = 1, or k = 2, or k = 3) — confirms the multi-scale fusion helps.
6. **w/o SMILES augmentation** — confirms augmentation helps.

We also include a **scaling study:** how performance changes as we vary training set size from 10% to 100%, to demonstrate data efficiency.

---

## 6. Limitations & Future Work

- **No 3D / morphology.** PolyChain uses SMILES only; it cannot capture chain conformation, entanglement, or crystalline morphology, which dominate mechanical properties and Tm in practice. We propose extending PECGN with an *optional* 3D branch that consumes MMFF-optimized coordinates (as in Uni-Poly's SchNet) for properties where this is known to help.
- **No copolymer sequence modeling.** Our copolymer support is *set-pooling* of monomers, which discards sequence. A future extension could integrate **BigSMILES** tokens and apply a sequence-level transformer to the per-monomer HAMF outputs.
- **Single property per training run.** Our fine-tuning is single-task; a multi-task head (à la POLYMERGNN) is straightforward to add but may hurt properties with scarce data.
- **Process conditions ignored.** Heating rate, thermal history, and humidity significantly affect measured Tg and Tm. We model only the structural input; a future "context token" could be added at test time if these are provided.
- **Long-range interactions.** PECGN's trimer proxy is a heuristic; ab initio chain effects (15+ repeats) are not modeled. A hierarchical extension with deeper oligomers is left for future work, gated by GPU memory.

---

## 7. Conclusion

PolyChain addresses three structural limitations that the four reference papers expose: (i) the monomer-only view of POLYMERGNN and Park et al., (ii) the rigid periodic featurization of Antoniuk et al., and (iii) the scale-uniform multimodal fusion of Uni-Poly. By introducing two genuinely new components — **HAMF** (cross-attention across oligomer scales) and **PECGN** (a learned, equivariant periodic boundary operator) — plus a polymer-specific SSL pretext (asterisk-mask reconstruction), PolyChain offers a competitive, SMILES-only architecture that is trainable from competition data, deployable in low-resource settings, and principled in its treatment of polymer periodicity. We expect it to yield the largest gains on properties where chain-scale phenomena matter most — Tm, Er, and density — and to provide a clean, interpretable architecture suitable for both deployment and future extensions to copolymer sequence, 3D conformation, and multi-task training.

---

## 8. References

 1. Queen, O., McCarver, G. A., Thatigotla, S., Abolins, B. P., Brown, C. L., Maroulas, V. & Vogiatzis, K. D. *Polymer graph neural networks for multitask property learning.* npj Computational Materials **9**, 90 (2023).
 2. Park, J. et al. *Prediction and interpretation of polymer properties using the graph convolutional network.* ACS Polymer Au **2**(4), 213–222 (2022).
 3. Antoniuk, E. R., Li, P., Kailkhura, B. & Hiszpanski, A. M. *Representing polymers as periodic graphs with learned descriptors for accurate polymer property predictions.* J. Chem. Inf. Model. **62**(22), 5435–5445 (2022).
 4. Huang, Q., Li, Y., Zhu, L., Zhao, Q. & Yu, W. *Uni-Poly: a unified multimodal multidomain polymer representation for property prediction.* npj Computational Materials **11**, 153 (2025).
 5. Kipf, T. N. & Welling, M. *Semi-supervised classification with graph convolutional networks.* ICLR (2017).
 6. Xu, K., Hu, W., Leskovec, J. & Jeglka, S. *How powerful are graph neural networks?* ICLR (2018).
 7. Velickovic, P. et al. *Graph attention networks.* arXiv:1710.10903 (2017).
 8. Lin, T.-S. et al. *BigSMILES: a structurally-based line notation for describing macromolecules.* ACS Cent. Sci. **5**, 1523–1531 (2019).
 9. Kuenneth, C. & Ramprasad, R. *polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics.* Nat. Commun. **14**, 4099 (2023).
10. Xu, C., Wang, Y. & Barati Farimani, A. *TransPolymer: a transformer-based language model for polymer property predictions.* npj Comput. Mater. **9**, 64 (2023).
11. Tao, L., Chen, G. & Li, Y. *Machine learning discovery of high-temperature polymers.* Patterns **2**, 100225 (2021).
12. Otsuka, S., Kuwajima, I., Hosoya, J., Xu, Y. & Yamazaki, M. *PoLyInfo: polymer database for polymeric materials design.* In Proc. 2011 Int. Conf. on Emerging Intelligent Data and Web Technologies 22–29 (IEEE, 2011).
13. Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. *A simple framework for contrastive learning of visual representations (InfoNCE).* ICML (2020).
14. Aldeghi, M. & Coley, C. W. *A graph representation of molecular ensembles for polymer property prediction.* Chem. Sci. **13**, 10486 (2022).
15. Schütt, K. et al. *SchNet: a continuous-filter convolutional neural network for modeling quantum interactions.* NeurIPS (2017).

---

## Repository Structure

```
PolyChain/
├── README.md                    # This file
├── docs/
│   ├── architecture.md          # Detailed architecture description
│   ├── training.md              # Training procedure
│   └── ablations.md             # Ablation study results
├── data/
│   ├── raw/                     # Original PoLyInfo + competition data
│   ├── processed/               # Tokenized SMILES, graph caches
│   └── README.md                # Data description
├── src/
│   ├── graphs/                  # Monomer/dimer/trimer/periodic graph builders
│   ├── models/
│   │   ├── gnn_backbone.py      # GIN-S encoder
│   │   ├── hamf.py              # Hierarchy-Aware Multi-Scale Fusion
│   │   ├── pecgn.py             # Periodic Equivariant Chain-Growth Network
│   │   └── polychain.py         # End-to-end model
│   ├── pretraining/             # Asterisk-mask + sub-SMILES SSL
│   ├── finetuning/              # Property prediction heads
│   └── evaluation/              # Metrics and baseline comparisons
├── configs/                     # YAML configs for ablations
├── scripts/
│   ├── pretrain.py
│   ├── finetune.py
│   └── evaluate.py
├── tests/                       # Unit tests for graph construction
├── notebooks/                   # Exploratory analysis
└── results/                     # Experiment outputs, plots
```

---

## Quick Start

> ⚠️ **Note:** This repository is currently a research proposal. Implementation code is planned for Week 1 of the timeline.

```bash
# Clone the repository
git clone https://github.com/your-org/polychain.git
cd polychain

# Create environment (Python 3.10+)
conda create -n polychain python=3.10
conda activate polychain
pip install -r requirements.txt

# (Planned) Pretrain on unlabeled SMILES
python scripts/pretrain.py --config configs/pretrain_default.yaml

# (Planned) Fine-tune on labeled property data
python scripts/finetune.py --config configs/finetune_tg.yaml

# (Planned) Run 5-fold cross-validation
python scripts/evaluate.py --config configs/cv_5fold.yaml
```

---

## Citation

If you use PolyChain in your research, please cite:

```bibtex
@misc{polychain2026,
  title={PolyChain: Hierarchical Periodic Transformer with Equivariant Multi-Scale Graph Reasoning for Polymer Property Prediction},
  author={PolyChain Research Team},
  year={2026},
  note={Research proposal, ML4Materials Workshop}
}
```

---

## License

This research proposal is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work builds on the insights from four foundational papers in polymer informatics:

- Queen et al. (2023) — multitask GNN design
- Park et al. (2022) — interpretable GCN-Attention
- Antoniuk et al. (2022) — periodic graph featurization
- Huang et al. (2025) — multimodal polymer representation

We thank the polymer informatics community for making PoLyInfo and related datasets publicly available.

---

*For questions or collaboration inquiries, please open an issue on GitHub.*