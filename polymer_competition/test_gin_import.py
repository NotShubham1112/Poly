import sys, warnings
warnings.filterwarnings("ignore")

from models.gnn import GINRegressor, GINEncoder
print("GINRegressor imported OK")

from features.graphs import smiles_to_graph
print("smiles_to_graph imported OK")

# Test with a simple molecule
g = smiles_to_graph("CCO")
print(f"Graph: {g}")
print(f"  x shape: {g.x.shape}")
print(f"  edge_index shape: {g.edge_index.shape}")
print(f"  edge_attr shape: {g.edge_attr.shape}")

# Test with polymer SMILES
g2 = smiles_to_graph("*CC*")
print(f"Polymer graph: {g2}")
print(f"  x shape: {g2.x.shape}")

# Test model creation
model = GINRegressor(in_dim=g.x.size(1), edge_dim=g.edge_attr.size(1),
                     hidden_dim=64, embed_dim=32, n_layers=2)
print(f"Model: {model}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
print("ALL OK")
