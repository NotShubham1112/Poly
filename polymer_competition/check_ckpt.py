import torch
for i in range(5):
    ckpt = torch.load(f"outputs/gin/tg/checkpoints/gin_gin_fold{i}_best.pt", map_location="cpu", weights_only=False)
    ms = ckpt["model_state"]
    h = ms["encoder.atom_encoder.weight"].shape[0]
    e = ms["encoder.output_proj.weight"].shape[0]
    print(f"Fold {i}: hidden_dim={h}, embed_dim={e}")
