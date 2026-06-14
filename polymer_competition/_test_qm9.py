"""
Minimal example: download QM9, preprocess, load into Dataset.
"""
import sys, os, warnings, logging
warnings.filterwarnings("ignore")
os.environ["RDKIT_SKIP_VALIDATION_WARNINGS"] = "1"

sys.path.insert(0, ".")

# 1. Download
print("=" * 55)
print("Step 1: Download QM9")
print("=" * 55)
from data.download_qm9 import download_qm9
df = download_qm9("data/")
print(f"  Columns: {list(df.columns)}")

# 2. Build SELFIES tokenizer
print("\n" + "=" * 55)
print("Step 2: Build SELFIES tokenizer")
print("=" * 55)
from models.generator.tokenizer import SELFIESTokenizer
tokenizer = SELFIESTokenizer()
smiles_sample = df["canonical_smiles"].dropna().sample(n=5000, random_state=42).tolist()
tokenizer.build_vocabulary(smiles_sample)
print(f"  Vocabulary size: {tokenizer.vocab_size}")

# 3. Preprocess and create datasets
print("\n" + "=" * 55)
print("Step 3: Preprocess + create datasets")
print("=" * 55)
from data.qm9_dataset import prepare_qm9_data
datasets, scaler = prepare_qm9_data(
    tokenizer=tokenizer,
    data_dir="data/",
    build_graphs=True,
    max_len=256,
    seed=42,
)
train_ds, val_ds, test_ds = datasets

# 4. Inspect a sample
print("\n" + "=" * 55)
print("Step 4: Sample inspection")
print("=" * 55)
sample = train_ds[0]
print(f"  SMILES      : {sample['smiles']}")
print(f"  input_ids   : {sample['input_ids'].shape}  {sample['input_ids'].dtype}")
print(f"  property    : {sample['property'].shape}  {sample['property']}")
print(f"  graph_sample: {type(sample.get('graph_sample')).__name__}")
if "graph_sample" in sample:
    gs = sample["graph_sample"]
    print(f"    monomer.x : {gs.monomer.x.shape}")
    print(f"    dimer.x   : {gs.dimer.x.shape}")

# 5. Batch collation
print("\n" + "=" * 55)
print("Step 5: Batch collation")
print("=" * 55)
from torch.utils.data import DataLoader
from training.train_generator import collate_generator  # reuse project's collate
batch_size = 4
loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    collate_fn=collate_generator)
batch = next(iter(loader))
print(f"  input_ids   : {batch['input_ids'].shape}")
print(f"  target_ids  : {batch['target_ids'].shape}")
print(f"  property    : {batch['property'].shape if 'property' in batch else 'N/A'}")
print(f"  graph_batch : {batch.get('graph_batch', 'N/A')}")
if "graph_batch" in batch:
    print(f"    monomer.x : {batch['graph_batch']['monomer'].x.shape}")
    print(f"    monomer.edge_index : {batch['graph_batch']['monomer'].edge_index.shape}")

# 6. Inverse-transform properties
print("\n" + "=" * 55)
print("Step 6: Inverse transform properties")
print("=" * 55)
raw = scaler.inverse(sample["property"].numpy().reshape(1, -1))
print(f"  Normalized : {sample['property'].numpy()}")
print(f"  Raw        : {raw[0]}")

print("\nAll done!")
