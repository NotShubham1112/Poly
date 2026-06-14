# Generative Polymer Chemistry — Implementation Scope

## Overview

Add a generative chemistry model that produces valid polymer SMILES strings,
optionally conditioned on a target property. This is a decoder-only or
encoder-decoder Transformer trained autoregressively on the existing
SMILES dataset.

---

## Architecture: Encoder-Decoder (Recommended)

```
Input (SMILES) → SELFIES → Token IDs
                    │
         ┌──────────┴──────────┐
         │                     │
    Graph Encoder         Property Target
    (PolyChain)           (optional scalar)
         │                     │
         └──────────┬──────────┘
                    │
            Cross-Attention
                    │
            Transformer Decoder
               (autoregressive)
               ┌────┴────┐
               │         │
         Token Head  Property Head (auxiliary)
               │         │
          Token Logits   pred_property
               │
          SELFIES → SMILES
               │
        Hierarchical Validator
         (post-hoc, metrics only)
               │
          Valid Molecules
```

### Why encoder-decoder?
- Reuses existing PolyChain graph encoder
- Enables property-conditional generation (target property → decoder conditioning)
- Graph provides structural priors the decoder can attend to

---

## Files to Create

### 1. `models/generator/tokenizer.py`

**Responsibility:** Convert SELFIES to/from token ID sequences.

```python
class SELFIESTokenizer:
    def __init__(self, vocab_path: str | None = None)

    def build_vocabulary(self, smiles_list: list[str]) -> None
        """Convert all SMILES → SELFIES, collect unique tokens."""

    def encode(self, smiles: str) -> torch.LongTensor
        """smiles → selfies → token_ids (includes <BOS>, <EOS>)."""

    def decode(self, token_ids: list[int]) -> str
        """token_ids → selfies → smiles."""

    @property
    def vocab_size(self) -> int
    @property
    def pad_token_id(self) -> int
    @property
    def bos_token_id(self) -> int
    @property
    def eos_token_id(self) -> int
```

**Design notes:**
- Use `selfies` library (`sf.encoder`, `sf.decoder`)
- Special tokens: `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<MASK>=3`
- Vocabulary size typically 30–60 for SELFIES
- Save/load vocabulary as JSON

---

### 2. `models/generator/transformer.py`

**Responsibility:** Decoder-only or encoder-decoder Transformer.

```python
class GeneratorConfig:
    vocab_size: int
    d_model: int = 512
    n_head: int = 8
    n_layer: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 256
    use_graph_encoder: bool = True   # False = decoder-only
    graph_dim: int = 256             # PolyChain hidden_dim


class GraphEncoder(nn.Module):
    """Wraps PolyChain's backbone; outputs graph-level embedding."""
    def __init__(self, polychain_config: dict)

    def forward(self, batch: dict) -> torch.Tensor
        """Returns (batch, graph_dim) graph embedding."""


class GeneratorDecoder(nn.Module):
    """Transformer decoder with cross-attention to graph embedding."""
    def __init__(self, config: GeneratorConfig)

    def forward(self, tokens: LongTensor,
                graph_emb: Tensor | None = None,
                property_cond: Tensor | None = None) -> tuple[Tensor, Tensor]
        """Returns (token_logits, decoder_hidden) for loss computation."""
```

**Design notes:**
- Causal masking for autoregressive generation
- Cross-attention to graph embedding — expand single vector into a **pseudo-sequence** (`Linear(1, seq_len)` + positional encoding), giving the decoder a structured conditional signal instead of a single vector
- Optional property conditioning via adaptive layernorm or concatenation
- Positional encoding: learned or rotary (RoPE preferred)

---

### 3. `models/generator/loss.py`

**Responsibility:** Clean, differentiable loss — no RDKit in the loss graph.

```python
class GenerativeLoss(nn.Module):
    def __init__(self, vocab_size: int,
                 property_weight: float = 0.5,
                 label_smoothing: float = 0.0)

    def forward(self, token_logits: Tensor, targets: Tensor,
                pred_property: Tensor | None = None,
                true_property: Tensor | None = None) -> Tensor
        """
        token_logits: (B, seq_len, vocab_size)
        targets:      (B, seq_len)

        L = CE(token_logits, targets)
          + λ_property * MSE(pred_property, true_property)  # optional

        NO RDKit penalties — SELFIES guarantees syntactic validity
        by construction. Validation is post-hoc (metrics only).
        """
```

**Design notes:**
- Cross-entropy with optional label smoothing
- Property MSE is the **latent alignment objective** — forces graph embedding to encode property-relevant info
- Property head: `MLP(graph_embedding) → scalar`
- Typical weight: λ_property=0.5 (tunable)
- No `validity_penalty`, no `valence_penalty` — those are non-differentiable and destabilize gradients

---

### 4. `models/generator/masking.py`

**Responsibility:** Lightweight token filtering for SELFIES generation.

```python
class SELFIESMask:
    """SELFIES is valid by construction — only lightweight inference-time masking."""

    def __init__(self, tokenizer: SELFIESTokenizer)

    def apply(self, logits: Tensor, prefix_tokens: list[int]) -> Tensor
        """
        Minimal constraints:
          - Prevent <PAD> before <EOS> (no premature padding)
          - Prevent <BOS> recurrence (once started, BOS is invalid)
          - Enforce max sequence length via <EOS> forcing
        No chemistry-heavy masking — SELFIES handles syntax, valence,
        ring closure, and bond validity automatically.
        """
```

**Design notes:**
- SELFIES grammar guarantees: syntax, valence, ring closure, bracket balance
- No need for `current_valence`, `open_rings`, or `adjacency` tracking
- Mask enforces: EOS at max length, no BOS recurrence, no premature padding, rare invalid token suppression
- **Minimal but not trivial** — structural discipline is still required for well-formed sequences
- Full chemical validation happens post-hoc in `MoleculeValidator` (metrics only)

---

### 5. `models/generator/validator.py`

**Responsibility:** Post-hoc validation for metrics only. NOT used in loss computation.

```python
class MoleculeValidator:
    """5-stage validation pipeline — metrics only, no training feedback."""

    def validate(self, smiles: str) -> tuple[bool, str]
        """
        Returns (is_valid, reason_if_invalid).

        Stages:
          1. Syntax: MolFromSmiles(smiles) is not None
          2. Sanitize: Chem.SanitizeMol(mol)
          3. Valence: all atoms have explicit_valence <= total_valence
          4. Kekulize: Chem.Kekulize(mol)
          5. Graph consistency: no duplicate edges, no disconnected fragments
        """
```

**Design notes:**
- Strictly post-hoc — never called during gradient computation
- Used for: metrics logging, dataset quality tracking, final filtering
- Returns detailed reason for debugging

---

### 6. `models/generator/curriculum.py`

**Responsibility:** Schedule training data by chemically-grounded complexity descriptors.

```python
class CurriculumScheduler:
    """
    CUMULATIVE phases — phase 0 ⊂ phase 1 ⊂ phase 2 ⊂ ... ⊂ phase 5.
    Each phase adds more complex molecules while retaining all previous data.
    This prevents forgetting of simple patterns.

    Phases defined by deterministic molecular descriptors:
      0: ring_count=0 AND heteroatom_count=0 AND heavy_atom_count≤6
         (simple alkanes — no heteroatoms, no rings)
      1: ring_count≤1 AND heteroatom_count=0
         (single rings, aromatics — still no heteroatoms)
      2: heteroatom_count≤2 AND ring_count≤1
         (+ O, N, halogens, up to 2)
      3: heavy_atom_count≤15 AND ring_count≤3  
         (+ branches, multiple rings, larger molecules)
      4: polymer detected (connection points '*' present)
         (+ repeat units, monomer patterns)
      5: full dataset (no filter)
    """

    def __init__(self, df: pd.DataFrame, n_phases: int = 6)

    def get_subset(self, phase: int) -> pd.DataFrame
        """Filter dataset to current phase by chemical descriptors."""

    def get_descriptors(self, smiles: str) -> dict
        """
        Returns deterministic fingerprint:
          ring_count:        number of rings (RDKit)
          heavy_atom_count:  number of non-H atoms
          heteroatom_count:  count of atoms not C or *
          branching_index:   max tree degree
          is_polymer:        '*' in SMILES
        """
```

**Design notes:**
- Each phase runs for N epochs (or until validation loss plateaus)
- Descriptors are deterministic — same SMILES always maps to same phase
- Prevents model from memorizing invalid patterns on complex examples before basics are learned

---

### 7. `training/train_generator.py`

**Responsibility:** Full training loop for the generator.

```
Usage:
    python -m training.train_generator \
        --epochs 100 \
        --batch_size 32 \
        --lr 1e-4 \
        --curriculum \
        --model_type encoder_decoder
```

```python
def train_epoch(model, loader, optimizer, loss_fn,
                scheduled_sampling_ratio: float):
    """
    scheduled_sampling_ratio: fraction of batches in this epoch that use
    free-run instead of teacher forcing. Set per-epoch (not per-step) for
    smooth convergence. Linearly annealed from 0.0 → 0.3 across training.
    """
    use_free_run = False  # same decision for entire batch
    for step, batch in enumerate(loader):
        # Set free-run flag at batch boundary (not per-step)
        if step == 0:
            use_free_run = random.random() < scheduled_sampling_ratio

        # Teacher forcing path
        tokens = batch["input_ids"]        # (B, seq_len)
        targets = batch["target_ids"]       # (B, seq_len), shifted right
        logits, decoder_hidden = model(tokens, return_hidden=True)

        # Property head — use decoder CLS token, not raw graph_emb
        cls_token = decoder_hidden[:, 0, :]  # CLS token
        pred_property = model.property_head(cls_token)

        # Primary loss: CE + property MSE
        loss = loss_fn(token_logits=logits, targets=targets,
                       pred_property=pred_property,
                       true_property=batch["property"])

        # Scheduled sampling — batch-level, not per-step
        if use_free_run:
            prefix = tokens[:, :1]        # <BOS> only
            model_out = model.generate(prefix, max_len=tokens.size(1))
            loss += 0.3 * F.cross_entropy(model_out.logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

**Design notes:**
- Teacher forcing with linear-annealed scheduled sampling (0% → 30%, never ≥50%)
- Ratio is computed **per epoch**, not per step — avoids high-variance per-step noise on 902-sample dataset
- Property head uses **decoder CLS token** (or mean pooled decoder states), not raw graph_emb — decoder hidden states carry richer signal for property alignment
- Gradient clipping (1.0) prevents instability with small dataset (902 samples)
- Curriculum phase advances when validation perplexity plateaus
- Checkpoint saves: `{person}_generator_phase{phase}_best.pt`
- Logs: loss, perplexity, validity%, uniqueness%, novelty%, property RMSE

---

### 8. `inference/generator.py`

**Responsibility:** Generate novel polymer SMILES from trained model.

```python
class PolymerGenerator:
    def __init__(self, checkpoint_path: str, device: str = "cpu")

    def generate(self,
                 n_samples: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 40,
                 top_p: float = 0.9,
                 max_length: int = 256,
                 property_target: float | None = None,
                 filter_valid: bool = True) -> list[dict]
        """
        Returns list of:
          [{"smiles": "...", "selfies": "...",
            "valid": True, "property_pred": 3.14,
            "log_prob": -12.5}, ...]
        """
```

**Sampling strategies:**
- Temperature scaling (T=1.0 default, lower = more deterministic)
- Top-k filtering (keep top 40 logits)
- Top-p (nucleus) sampling (keep cumulative probability mass p)
- Beam search (width=5) for optimal sequences
- All samples passed through `MoleculeValidator`; invalid ones re-sampled or discarded

---

## Integration with Existing Code

| Existing Component | How Generator Uses It |
|---|---|
| `features/graphs.py` | Graph encoder in encoder-decoder variant |
| `models/polychain/polychain.py` | Backbone for graph encoder (shared weights or frozen) |
| `models/polychain/cst.py` | CST features as conditioning input |
| `training/train.py` | Pattern for training loops, checkpoint save/load |
| `config.yaml` | Extend with `generator:` section |
| `inference/predictor.py` | Not directly — use `inference/generator.py` instead |

---

## Timeline Estimate

| Phase | Tasks | Est. Time |
|---|---|---|
| 1 | Tokenizer + vocabulary builder (SELFIES) | 0.5 day |
| 2 | Transformer decoder + property head | 1 day |
| 3 | Training loop (teacher forcing + linear annealing) | 0.5 day |
| 4 | Lightweight SELFIES mask + post-hoc validator | 0.5 day |
| 5 | Graph encoder integration (cross-attention) | 0.5 day |
| 6 | Curriculum scheduler (RDKit descriptors) | 0.5 day |
| 7 | Sampling (top-k, top-p, beam search) | 0.5 day |
| 8 | Testing + notebook integration | 0.5 day |
| **Total** | | **~4.5 days** |

---

## Key Design Decisions (Corrected)

1. **SELFIES over SMILES for tokenization.** Guarantees syntactic validity by construction. Eliminates ~70% of RDKit crashes at generation time. With SELFIES, the masking layer becomes lightweight (no chemistry-heavy valence/ring/bond tracking).

2. **Encoder-decoder with latent alignment.** Reuses existing PolyChain graph encoder; enables property-conditional generation. Property head uses **decoder CLS token** (not raw graph embedding) — decoder hidden states carry richer signal. Cross-attention expands graph embedding into a **pseudo-sequence** (`Linear(1, seq_len)` + positional encoding) instead of conditioning on a single vector.

3. **Loss: CE + property MSE only.** No RDKit-based penalties in the loss graph. `validity_penalty` and `valence_penalty` are non-differentiable and destabilize gradients. SELFIES guarantees syntactic validity; RDKit runs post-hoc for metrics only.

4. **Scheduled sampling with linear annealing.** Starts at 0% free-run, linearly anneals to 30% max (never ≥50%). Ratio is **per-epoch**, not per-step — batch-level scheduling avoids high-variance training noise on small datasets (902 samples).

5. **Curriculum training with chemically-grounded descriptors.** Phases partitioned by deterministic RDKit descriptors (`ring_count`, `heteroatom_count`, `heavy_atom_count`, `branching_index`, `is_polymer`). **Cumulative** (phase 0 ⊂ phase 1 ⊂ ...) to prevent forgetting of simple patterns.

6. **RDKit as validator, not generator gate.** RDKit runs post-hoc for metrics and final quality filtering. Generation constraints are handled by SELFIES grammar + lightweight token discipline mask.
