import pandas as pd
import urllib.request
import os

def download_and_prepare_sample_data():
    """
    Downloads the public ESOL (Delaney) dataset and formats it as train.csv and test.csv
    to test the PolyChain pipeline.
    """
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    print("Downloading sample dataset (ESOL)...")
    
    # Download the CSV
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(data_dir, "esol_raw.csv")
    urllib.request.urlretrieve(url, raw_path)
    
    # Load and format
    df = pd.read_csv(raw_path)
    
    # ESOL columns: "smiles", "measured log solubility in mols per litre"
    # We need: id, SMILES, property
    df = df[['smiles', 'measured log solubility in mols per litre']].copy()
    df.columns = ['SMILES', 'property']
    df['id'] = [f"sample_{i}" for i in range(len(df))]
    
    # Reorder columns
    df = df[['id', 'SMILES', 'property']]
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train (80%) and test (20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    
    # Test set shouldn't have the target property in a real competition,
    # but our pipeline might not care. We'll drop it to mimic competition.
    test_df = df.iloc[split_idx:].copy()
    test_df = test_df[['id', 'SMILES']]
    
    # Save
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    
    # Cleanup raw file
    os.remove(raw_path)
    
    print(f"Created train.csv ({len(train_df)} rows) and test.csv ({len(test_df)} rows) in data/")

if __name__ == "__main__":
    download_and_prepare_sample_data()
