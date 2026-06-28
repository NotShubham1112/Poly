"""
features package – molecular representation logic.

Modules:
    fingerprints   : Morgan, MACCS, atom-pair fingerprints
    descriptors    : RDKit 200+ descriptors, topological, physicochemical
    graphs         : Base graph builders (monomer, k-mer, periodic) shared by all GNNs
    custom_polymer : Polymer-specific features (asterisks, rigidity, H-bond density)
    build_features : Master function that merges everything into a feature matrix
    graph_utils    : Helpers for multi-scale graph construction (shared with PolyChain)
    target_transforms : Box-Cox, quantile, and log transforms for targets
"""
