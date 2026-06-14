"""
models package – model definitions (architecture only, no training logic).

Submodules:
    baselines          : Linear/Ridge/Lasso wrappers
    tree_models        : RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost
    mlp                : Descriptor & fingerprint MLPs
    gnn                : GCN, GAT, MPNN (used as baselines)
    graph_transformer  : Graph Transformer with TransformerConv
    chemberta          : ChemBERTa embedding extractor + prediction head
    fusionnet          : PolymerFusionNet (multimodal cross-attention)
    polychain          : PolyChain – Hierarchical Periodic Transformer (★)
    generator          : Generative polymer chemistry (SELFIES tokenizer, decoder, etc.)
"""
