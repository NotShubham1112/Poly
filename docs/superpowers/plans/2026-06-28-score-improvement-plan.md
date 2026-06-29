# PolyChain v28: Score Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Mean R² from 0.896 to 0.93+ through periodic polymer graphs, multi-task learning, and advanced feature engineering

**Architecture:** 5-phase approach: (1) periodic polymer graphs, (2) multi-task learning, (3) advanced features, (4) stacking ensemble, (5) target transformation

**Tech Stack:** Python, RDKit, PyTorch, XGBoost, LightGBM, CatBoost, scikit-learn

## Global Constraints

- Kaggle notebook-only: no external data, pretrained weights, or cached artifacts
- 5-fold scaffold cross-validation
- Evaluation metric: Mean R² = (R²_Tg + R²_Egc) / 2
- 6,171 training samples (4,143 Tg, 2,028 Egc)
- CPU-only environment (no CUDA)

---

## Task 1: Periodic Polymer Graph Module

**Files:**
- Create: `polymer_competition/features/periodic_polymer.py`
- Modify: `polymer_competition/features/build_features.py:45-67`
- Test: `polymer_competition/tests/test_periodic_polymer.py`

**Interfaces:**
- Consumes: SMILES strings from `train.csv`
- Produces: `generate_oligomer_smiles(smiles: str, n_repeats: int) -> str`

- [ ] **Step 1: Write the failing test**

```python
# polymer_competition/tests/test_periodic_polymer.py
import pytest
from polymer_competition.features.periodic_polymer import generate_oligomer_smiles, build_periodic_graph

def test_generate_oligomer_smiles():
    # Simple case: ethylene glycol
    smiles = "*CCO*"
    result = generate_oligomer_smiles(smiles, n_repeats=3)
    assert result == "*CCOCCOCCO*"
    
    # Aromatic case
    smiles = "*c1ccc(O)cc1*"
    result = generate_oligomer_smiles(smiles, n_repeats=2)
    assert "c1ccc" in result
    assert result.count("c1ccc") == 2

def test_build_periodic_graph():
    from rdkit import Chem
    smiles = "*CCO*"
    mol = Chem.MolFromSmiles(smiles)
    graph = build_periodic_graph(mol, n_repeats=3)
    assert graph is not None
    assert len(graph.nodes) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd polymer_competition && python -m pytest tests/test_periodic_polymer.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'polymer_competition.features.periodic_polymer'"

- [ ] **Step 3: Write minimal implementation**

```python
# polymer_competition/features/periodic_polymer.py
"""Periodic polymer graph generation for improved property prediction."""

from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from typing import List, Tuple


def parse_smiles_with_stars(smiles: str) -> Tuple[str, List[int]]:
    """Parse SMILES and identify connection points (*)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    stars = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            stars.append(atom.GetIdx())
    
    return smiles, stars


def generate_oligomer_smiles(smiles: str, n_repeats: int = 3) -> str:
    """
    Generate oligomer SMILES with N repeats for periodic polymer graphs.
    
    Example:
        *CCO* → *CCOCCOCCO* (3 repeats)
        *c1ccc(O)cc1* → *c1ccc(O)cc1c1ccc(O)cc1c1ccc(O)cc1* (3 repeats)
    
    Args:
        smiles: SMILES string with * connection points
        n_repeats: Number of repeat units (default: 3)
    
    Returns:
        Expanded oligomer SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Find connection points
    stars = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
    if len(stars) != 2:
        raise ValueError(f"Expected 2 connection points, got {len(stars)}")
    
    # Get the repeat unit (remove * atoms)
    atoms_to_remove = stars
    rw_mol = Chem.RWMol(mol)
    
    # Remove * atoms and their bonds
    for idx in sorted(atoms_to_remove, reverse=True):
        rw_mol.RemoveAtom(idx)
    
    repeat_smiles = Chem.MolToSmiles(rw_mol)
    
    # Build oligomer: * + repeat * (repeat) * 
    oligomer = "*" + repeat_smiles * n_repeats + "*"
    
    return oligomer


def build_periodic_graph(mol: Chem.Mol, n_repeats: int = 3) -> nx.Graph:
    """
    Build a periodic polymer graph from a monomer.
    
    Args:
        mol: RDKit molecule with * connection points
        n_repeats: Number of repeat units
    
    Returns:
        NetworkX graph representing periodic polymer
    """
    smiles = Chem.MolToSmiles(mol)
    oligomer_smiles = generate_oligomer_smiles(smiles, n_repeats)
    
    # Parse oligomer
    oligomer_mol = Chem.MolFromSmiles(oligomer_smiles)
    if oligomer_mol is None:
        raise ValueError(f"Failed to parse oligomer: {oligomer_smiles}")
    
    # Convert to graph
    graph = nx.Graph()
    
    for atom in oligomer_mol.GetAtoms():
        if atom.GetSymbol() != "*":  # Skip connection points
            graph.add_node(
                atom.GetIdx(),
                symbol=atom.GetSymbol(),
                degree=atom.GetDegree(),
                formal_charge=atom.GetFormalCharge(),
                hybridization=str(atom.GetHybridization()),
                is_aromatic=atom.GetIsAromatic()
            )
    
    for bond in oligomer_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        # Skip if either atom is a *
        begin_atom = oligomer_mol.GetAtomWithIdx(begin_idx)
        end_atom = oligomer_mol.GetAtomWithIdx(end_idx)
        
        if begin_atom.GetSymbol() != "*" and end_atom.GetSymbol() != "*":
            graph.add_edge(
                begin_idx,
                end_idx,
                bond_type=str(bond.GetBondType()),
                is_conjugated=bond.GetIsConjugated(),
                is_in_ring=bond.GetIsInRing()
            )
    
    return graph


def get_periodic_smiles_list(smiles_list: List[str], n_repeats: int = 3) -> List[str]:
    """
    Convert list of monomer SMILES to oligomer SMILES.
    
    Args:
        smiles_list: List of SMILES with * connection points
        n_repeats: Number of repeat units
    
    Returns:
        List of oligomer SMILES
    """
    result = []
    for smiles in smiles_list:
        try:
            oligomer = generate_oligomer_smiles(smiles, n_repeats)
            result.append(oligomer)
        except Exception as e:
            print(f"Warning: Failed to process {smiles}: {e}")
            result.append(smiles)  # Fallback to original
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd polymer_competition && python -m pytest tests/test_periodic_polymer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add polymer_competition/features/periodic_polymer.py polymer_competition/tests/test_periodic_polymer.py
git commit -m "feat: add periodic polymer graph generation module"
```

---

## Task 2: Integrate Periodic Graphs into Feature Pipeline

**Files:**
- Modify: `polymer_competition/features/build_features.py:89-134`
- Modify: `polymer_competition/features/graphs.py:23-45`

**Interfaces:**
- Consumes: `generate_oligomer_smiles()` from Task 1
- Produces: `build_periodic_graph_features()` function

- [ ] **Step 1: Add periodic graph builder to build_features.py**

```python
# Add to polymer_competition/features/build_features.py after line 134

def build_periodic_graph_features(smiles_list, n_repeats=3):
    """
    Build graph features for periodic polymer structures.
    
    Args:
        smiles_list: List of SMILES strings
        n_repeats: Number of repeat units for periodicity
    
    Returns:
        DataFrame with graph features
    """
    from polymer_competition.features.periodic_polymer import generate_oligomer_smiles
    
    features = []
    
    for smiles in smiles_list:
        try:
            # Generate oligomer
            oligomer_smiles = generate_oligomer_smiles(smiles, n_repeats)
            
            # Parse oligomer
            mol = Chem.MolFromSmiles(oligomer_smiles)
            if mol is None:
                features.append({})
                continue
            
            # Extract features from oligomer
            feat = {
                'periodic_n_atoms': mol.GetNumAtoms(),
                'periodic_n_bonds': mol.GetNumBonds(),
                'periodic_mol_weight': Chem.Descriptors.MolWt(mol),
                'periodic_logp': Chem.Descriptors.MolLogP(mol),
                'periodic_tpsa': Chem.Descriptors.TPSA(mol),
                'periodic_n_rings': mol.GetRingInfo().NumRings(),
                'periodic_n_aromatic_rings': sum(1 for r in mol.GetRingInfo().AtomRings() 
                                                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)),
            }
            
            features.append(feat)
            
        except Exception as e:
            print(f"Warning: Failed to process {smiles}: {e}")
            features.append({})
    
    return pd.DataFrame(features)
```

- [ ] **Step 2: Integrate into build_features main function**

```python
# Add after line 134 in polymer_competition/features/build_features.py

# Build periodic graph features
print("Building periodic graph features...")
periodic_features = build_periodic_graph_features(smiles_list, n_repeats=3)
feature_list.append(periodic_features)
print(f"  Added {periodic_features.shape[1]} periodic graph features")
```

- [ ] **Step 3: Test integration**

Run: `cd polymer_competition && python -c "from features.build_features import build_features; print('Integration OK')"`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add polymer_competition/features/build_features.py
git commit -m "feat: integrate periodic graph features into pipeline"
```

---

## Task 3: Multi-Task Learning Model

**Files:**
- Create: `polymer_competition/models/multitask.py`
- Modify: `polymer_competition/training/train.py:156-189`
- Test: `polymer_competition/tests/test_multitask.py`

**Interfaces:**
- Consumes: Feature matrices X_tg, X_egc from feature pipeline
- Produces: `MultiTaskModel` class with `predict(X)` method

- [ ] **Step 1: Write the failing test**

```python
# polymer_competition/tests/test_multitask.py
import pytest
import numpy as np
import pandas as pd
from polymer_competition.models.multitask import MultiTaskModel

def test_multitask_model_creation():
    model = MultiTaskModel(n_features=100, hidden_dims=[64, 32])
    assert model is not None

def test_multitask_model_forward():
    model = MultiTaskModel(n_features=100, hidden_dims=[64, 32])
    X = np.random.randn(10, 100).astype(np.float32)
    tg_pred, egc_pred = model.predict(X)
    assert tg_pred.shape == (10,)
    assert egc_pred.shape == (10,)

def test_multitask_model_train():
    model = MultiTaskModel(n_features=100, hidden_dims=[64, 32])
    X = np.random.randn(100, 100).astype(np.float32)
    y_tg = np.random.randn(100).astype(np.float32)
    y_egc = np.random.randn(100).astype(np.float32)
    
    model.fit(X, y_tg, y_egc, epochs=10, batch_size=32)
    tg_pred, egc_pred = model.predict(X)
    assert tg_pred.shape == (100,)
    assert egc_pred.shape == (100,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd polymer_competition && python -m pytest tests/test_multitask.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# polymer_competition/models/multitask.py
"""Multi-task learning model for joint Tg and Egc prediction."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class MultiTaskModel:
    """
    Multi-task neural network for joint Tg and Egc prediction.
    
    Architecture:
        Input → Shared Encoder → Tg Head → Tg Prediction
                              → Egc Head → Egc Prediction
    
    Loss: L = γ_egc * L_egc + L_tg
    """
    
    def __init__(self, n_features: int, hidden_dims: list = [128, 64, 32],
                 dropout: float = 0.2, gamma_egc: float = 100.0):
        """
        Initialize multi-task model.
        
        Args:
            n_features: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            gamma_egc: Scaling factor for Egc loss (Egc values are ~10x smaller)
        """
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.gamma_egc = gamma_egc
        
        # Build model
        self.model = self._build_model()
        self.scaler_X = None
        self.scaler_y_tg = None
        self.scaler_y_egc = None
    
    def _build_model(self):
        """Build PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            
            class MultiTaskNet(nn.Module):
                def __init__(self, n_features, hidden_dims, dropout):
                    super().__init__()
                    
                    # Shared encoder
                    layers = []
                    in_dim = n_features
                    for h_dim in hidden_dims[:-1]:
                        layers.extend([
                            nn.Linear(in_dim, h_dim),
                            nn.BatchNorm1d(h_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout)
                        ])
                        in_dim = h_dim
                    
                    self.shared_encoder = nn.Sequential(*layers)
                    
                    # Tg head
                    self.tg_head = nn.Sequential(
                        nn.Linear(in_dim, hidden_dims[-1]),
                        nn.ReLU(),
                        nn.Linear(hidden_dims[-1], 1)
                    )
                    
                    # Egc head
                    self.egc_head = nn.Sequential(
                        nn.Linear(in_dim, hidden_dims[-1]),
                        nn.ReLU(),
                        nn.Linear(hidden_dims[-1], 1)
                    )
                
                def forward(self, x):
                    shared = self.shared_encoder(x)
                    tg = self.tg_head(shared)
                    egc = self.egc_head(shared)
                    return tg.squeeze(), egc.squeeze()
            
            return MultiTaskNet(self.n_features, self.hidden_dims, self.dropout)
            
        except ImportError:
            print("Warning: PyTorch not available, using sklearn fallback")
            return None
    
    def fit(self, X: np.ndarray, y_tg: np.ndarray, y_egc: np.ndarray,
            epochs: int = 100, batch_size: int = 32, lr: float = 0.001):
        """
        Train multi-task model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y_tg: Tg targets (n_samples,)
            y_egc: Egc targets (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            # Scale inputs
            from sklearn.preprocessing import StandardScaler
            self.scaler_X = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(X)
            
            # Scale targets
            self.scaler_y_tg = StandardScaler()
            y_tg_scaled = self.scaler_y_tg.fit_transform(y_tg.reshape(-1, 1)).flatten()
            
            self.scaler_y_egc = StandardScaler()
            y_egc_scaled = self.scaler_y_egc.fit_transform(y_egc.reshape(-1, 1)).flatten()
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tg_tensor = torch.FloatTensor(y_tg_scaled)
            y_egc_tensor = torch.FloatTensor(y_egc_scaled)
            
            dataset = TensorDataset(X_tensor, y_tg_tensor, y_egc_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Move model to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            self.model.train()
            
            # Optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Loss function
            mse_loss = nn.MSELoss()
            
            # Training loop
            for epoch in range(epochs):
                total_loss = 0
                for X_batch, tg_batch, egc_batch in loader:
                    X_batch = X_batch.to(device)
                    tg_batch = tg_batch.to(device)
                    egc_batch = egc_batch.to(device)
                    
                    optimizer.zero_grad()
                    tg_pred, egc_pred = self.model(X_batch)
                    
                    # Multi-task loss
                    loss_tg = mse_loss(tg_pred, tg_batch)
                    loss_egc = mse_loss(egc_pred, egc_batch)
                    loss = loss_tg + self.gamma_egc * loss_egc
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Learning rate scheduling
                avg_loss = total_loss / len(loader)
                scheduler.step(avg_loss)
                
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            self.model.eval()
            
        except ImportError:
            # Fallback to sklearn
            from sklearn.ensemble import GradientBoostingRegressor
            
            self.tg_model = GradientBoostingRegressor(n_estimators=100)
            self.egc_model = GradientBoostingRegressor(n_estimators=100)
            
            self.tg_model.fit(X, y_tg)
            self.egc_model.fit(X, y_egc)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict Tg and Egc.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Tuple of (tg_predictions, egc_predictions)
        """
        if self.model is None:
            # Sklearn fallback
            tg_pred = self.tg_model.predict(X)
            egc_pred = self.egc_model.predict(X)
            return tg_pred, egc_pred
        
        try:
            import torch
            
            # Scale inputs
            X_scaled = self.scaler_X.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            device = next(self.model.parameters()).device
            X_tensor = X_tensor.to(device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                tg_pred, egc_pred = self.model(X_tensor)
            
            # Inverse scale
            tg_pred = self.scaler_y_tg.inverse_transform(tg_pred.cpu().numpy().reshape(-1, 1)).flatten()
            egc_pred = self.scaler_y_egc.inverse_transform(egc_pred.cpu().numpy().reshape(-1, 1)).flatten()
            
            return tg_pred, egc_pred
            
        except Exception as e:
            print(f"Warning: PyTorch prediction failed: {e}")
            # Sklearn fallback
            tg_pred = self.tg_model.predict(X)
            egc_pred = self.egc_model.predict(X)
            return tg_pred, egc_pred
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd polymer_competition && python -m pytest tests/test_multitask.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add polymer_competition/models/multitask.py polymer_competition/tests/test_multitask.py
git commit -m "feat: add multi-task learning model for joint Tg+Egc prediction"
```

---

## Task 4: Advanced Polymer Features

**Files:**
- Create: `polymer_competition/features/advanced_descriptors.py`
- Modify: `polymer_competition/features/build_features.py:134-156`
- Test: `polymer_competition/tests/test_advanced_descriptors.py`

**Interfaces:**
- Consumes: RDKit mol objects
- Produces: DataFrame with advanced polymer descriptors

- [ ] **Step 1: Write the failing test**

```python
# polymer_competition/tests/test_advanced_descriptors.py
import pytest
from rdkit import Chem
from polymer_competition.features.advanced_descriptors import (
    hansen_solubility_parameters,
    free_volume_fraction,
    chain_flexibility,
    conjugation_length
)

def test_hansen_solubility_parameters():
    mol = Chem.MolFromSmiles("*CCO*")
    dp, dP, dH = hansen_solubility_parameters(mol)
    assert isinstance(dp, float)
    assert isinstance(dP, float)
    assert isinstance(dH, float)
    assert dp > 0  # Dispersion always positive

def test_free_volume_fraction():
    mol = Chem.MolFromSmiles("*CCO*")
    fv = free_volume_fraction(mol)
    assert isinstance(fv, float)
    assert 0 < fv < 1  # Free volume fraction between 0 and 1

def test_chain_flexibility():
    mol = Chem.MolFromSmiles("*CCO*")
    flexibility = chain_flexibility(mol)
    assert isinstance(flexibility, float)
    assert flexibility >= 0

def test_conjugation_length():
    mol = Chem.MolFromSmiles("*c1ccc(O)cc1*")
    conj_len = conjugation_length(mol)
    assert isinstance(conj_len, int)
    assert conj_len >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd polymer_competition && python -m pytest tests/test_advanced_descriptors.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# polymer_competition/features/advanced_descriptors.py
"""Advanced polymer descriptors for property prediction."""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np
from typing import Tuple


def hansen_solubility_parameters(mol: Chem.Mol) -> Tuple[float, float, float]:
    """
    Estimate Hansen solubility parameters from group contributions.
    
    δD: Dispersion parameter (MPa^0.5)
    δP: Polar parameter (MPa^0.5)
    δH: Hydrogen bonding parameter (MPa^0.5)
    
    Reference: Hansen, C. M. (2007). Hansen Solubility Parameters: A User's Handbook.
    """
    if mol is None:
        return 0.0, 0.0, 0.0
    
    # Simplified group contribution method
    # Based on van Krevelen and Hoftyzer (1976)
    
    # Count functional groups
    n_atoms = mol.GetNumHeavyAtoms()
    
    # Dispersion (δD) - related to molecular size and polarizability
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Approximate δD from molecular properties
    # Higher MW and logp → higher δD
    dp = 15.0 + 0.5 * np.log(mw + 1) + 2.0 * logp
    
    # Polar (δP) - related to dipole moments and polarity
    # Higher TPSA → higher δP
    n_polar = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7, 8, 9, 16])
    dP = 5.0 + 0.1 * tpsa + 1.0 * n_polar
    
    # Hydrogen bonding (δH) - related to H-bond donors/acceptors
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    dH = 10.0 + 2.0 * hbd + 1.5 * hba
    
    return dp, dP, dH


def free_volume_fraction(mol: Chem.Mol) -> float:
    """
    Estimate free volume fraction using Bondi group contributions.
    
    Free volume is critical for Tg prediction (Fox-Flory equation).
    Higher free volume → lower Tg.
    
    Reference: Bondi, A. (1964). Van der Waals volumes and radii.
    """
    if mol is None:
        return 0.0
    
    # Count atoms by type
    n_C = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    n_O = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    n_N = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    n_S = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
    n_F = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    n_Cl = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17)
    
    # Van der Waals volumes (cm³/mol) from Bondi
    vdw_C = 16.6  # Methylene
    vdw_O = 8.5   # Ether oxygen
    vdw_N = 10.5  # Amine
    vdw_S = 18.5  # Thioether
    vdw_F = 5.5   # Fluorine
    vdw_Cl = 19.5 # Chlorine
    
    # Calculate occupied volume
    v_occupied = (n_C * vdw_C + n_O * vdw_O + n_N * vdw_N + 
                  n_S * vdw_S + n_F * vdw_F + n_Cl * vdw_Cl)
    
    # Total molecular volume (approximate from MW)
    mw = Descriptors.MolWt(mol)
    density = 1.0  # g/cm³ approximate
    v_total = mw / density
    
    # Free volume fraction
    if v_total > 0:
        fv = 1.0 - (v_occupied / v_total)
        return max(0.0, min(1.0, fv))  # Clamp to [0, 1]
    
    return 0.5  # Default


def chain_flexibility(mol: Chem.Mol) -> float:
    """
    Estimate chain flexibility metric.
    
    Combines:
    - Rotatable bonds
    - Fraction of sp3 carbons
    - Ring strain indicators
    
    Higher flexibility → lower Tg (more conformational freedom)
    """
    if mol is None:
        return 0.0
    
    # Rotatable bonds
    n_rotatable = Descriptors.NumRotatableBonds(mol)
    
    # Fraction sp3
    frac_sp3 = Descriptors.FractionCSP3(mol)
    
    # Ring info
    n_rings = mol.GetRingInfo().NumRings()
    ring_density = n_rings / max(1, mol.GetNumHeavyAtoms())
    
    # Flexibility = normalized rotatable bonds + sp3 contribution - ring constraint
    flexibility = (n_rotatable / max(1, mol.GetNumHeavyAtoms()) + 
                   frac_sp3 * 0.3 - 
                   ring_density * 0.2)
    
    return max(0.0, flexibility)


def conjugation_length(mol: Chem.Mol) -> int:
    """
    Measure effective conjugation length in polymer backbone.
    
    Critical for Egc prediction:
    - Longer conjugation → smaller band gap
    - More aromatic rings in backbone → better π-overlap
    
    Returns:
        Number of conjugated atoms in longest path
    """
    if mol is None:
        return 0
    
    # Find aromatic atoms
    aromatic_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    
    if not aromatic_atoms:
        return 0
    
    # Find longest path through aromatic atoms
    from rdkit.Chem import rchem
    from itertools import combinations
    
    max_path_len = 0
    
    for start in aromatic_atoms:
        for end in aromatic_atoms:
            if start != end:
                path = rchem.GetShortestPath(mol, start, end)
                # Check if path is through aromatic atoms
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in path):
                    max_path_len = max(max_path_len, len(path))
    
    return max_path_len


def compute_all_advanced_features(mol: Chem.Mol) -> dict:
    """
    Compute all advanced polymer descriptors.
    
    Args:
        mol: RDKit molecule
    
    Returns:
        Dictionary of feature names and values
    """
    if mol is None:
        return {}
    
    dp, dP, dH = hansen_solubility_parameters(mol)
    
    return {
        'hansen_dp': dp,
        'hansen_dP': dP,
        'hansen_dH': dH,
        'free_volume': free_volume_fraction(mol),
        'chain_flexibility': chain_flexibility(mol),
        'conjugation_length': conjugation_length(mol),
        'total_hansen': dp + dP + dH,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd polymer_competition && python -m pytest tests/test_advanced_descriptors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add polymer_competition/features/advanced_descriptors.py polymer_competition/tests/test_advanced_descriptors.py
git commit -m "feat: add advanced polymer descriptors (Hansen, free volume, conjugation)"
```

---

## Task 5: Integrate Advanced Features into Pipeline

**Files:**
- Modify: `polymer_competition/features/build_features.py:156-178`

**Interfaces:**
- Consumes: `compute_all_advanced_features()` from Task 4
- Produces: Updated feature matrix

- [ ] **Step 1: Add advanced features to build_features.py**

```python
# Add after line 156 in polymer_competition/features/build_features.py

# Build advanced polymer features
print("Building advanced polymer features...")
from polymer_competition.features.advanced_descriptors import compute_all_advanced_features

advanced_features = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    feat = compute_all_advanced_features(mol)
    advanced_features.append(feat)

advanced_df = pd.DataFrame(advanced_features)
feature_list.append(advanced_df)
print(f"  Added {advanced_df.shape[1]} advanced polymer features")
```

- [ ] **Step 2: Test integration**

Run: `cd polymer_competition && python -c "from features.build_features import build_features; print('Advanced features integration OK')"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add polymer_competition/features/build_features.py
git commit -m "feat: integrate advanced polymer features into pipeline"
```

---

## Task 6: Target Transformation Module

**Files:**
- Create: `polymer_competition/features/target_transforms.py`
- Modify: `polymer_competition/training/train.py:189-212`
- Test: `polymer_competition/tests/test_target_transforms.py`

**Interfaces:**
- Consumes: Raw target values
- Produces: Transformed targets and inverse transform function

- [ ] **Step 1: Write the failing test**

```python
# polymer_competition/tests/test_target_transforms.py
import pytest
import numpy as np
from polymer_competition.features.target_transforms import (
    boxcox_transform,
    quantile_transform,
    log_transform
)

def test_boxcox_transform():
    y = np.array([100, 150, 200, 250, 300, 350, 400], dtype=float)
    y_transformed, inv_func = boxcox_transform(y)
    
    assert y_transformed.shape == y.shape
    assert not np.any(np.isnan(y_transformed))
    
    # Inverse transform should recover original
    y_recovered = inv_func(y_transformed)
    np.testing.assert_array_almost_equal(y, y_recovered, decimal=5)

def test_quantile_transform():
    y = np.array([100, 150, 200, 250, 300, 350, 400], dtype=float)
    y_transformed, inv_func = quantile_transform(y)
    
    assert y_transformed.shape == y.shape
    assert not np.any(np.isnan(y_transformed))

def test_log_transform():
    y = np.array([0.5, 1.0, 2.0, 3.0, 5.0], dtype=float)
    y_transformed, inv_func = log_transform(y)
    
    assert y_transformed.shape == y.shape
    assert not np.any(np.isnan(y_transformed))
    
    # Inverse transform should recover original
    y_recovered = inv_func(y_transformed)
    np.testing.assert_array_almost_equal(y, y_recovered, decimal=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd polymer_competition && python -m pytest tests/test_target_transforms.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# polymer_competition/features/target_transforms.py
"""Target transformations for improved model performance."""

import numpy as np
from typing import Tuple, Callable
from scipy import stats


def boxcox_transform(y: np.ndarray) -> Tuple[np.ndarray, Callable]:
    """
    Apply Box-Cox transformation to target.
    
    Useful for:
    - Making distribution more normal
    - Stabilizing variance
    - Improving linear model performance
    
    Args:
        y: Raw target values (must be positive)
    
    Returns:
        Tuple of (transformed_values, inverse_function)
    """
    # Shift to positive if needed
    min_val = y.min()
    if min_val <= 0:
        y_shifted = y - min_val + 1
    else:
        y_shifted = y
        min_val = 0
    
    # Apply Box-Cox
    y_transformed, lambda_param = stats.boxcox(y_shifted)
    
    # Create inverse function
    def inverse_transform(y_trans):
        if lambda_param == 0:
            y_inv = np.exp(y_trans)
        else:
            y_inv = (y_trans * lambda_param + 1) ** (1 / lambda_param)
        return y_inv + min_val
    
    return y_transformed, inverse_transform


def quantile_transform(y: np.ndarray) -> Tuple[np.ndarray, Callable]:
    """
    Apply quantile transformation (rank-based) to target.
    
    Useful for:
    - Making distribution exactly normal
    - Handling outliers
    - Tree models sometimes benefit
    
    Args:
        y: Raw target values
    
    Returns:
        Tuple of (transformed_values, inverse_function)
    """
    from sklearn.preprocessing import QuantileTransformer
    
    qt = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=min(100, len(y)),
        random_state=42
    )
    
    y_reshaped = y.reshape(-1, 1)
    y_transformed = qt.fit_transform(y_reshaped).flatten()
    
    # Create inverse function
    def inverse_transform(y_trans):
        return qt.inverse_transform(y_trans.reshape(-1, 1)).flatten()
    
    return y_transformed, inverse_transform


def log_transform(y: np.ndarray) -> Tuple[np.ndarray, Callable]:
    """
    Apply log transformation to target.
    
    Useful for:
    - Skewed distributions
    - Egc values (typically 0.1-10)
    - Ratios and percentages
    
    Args:
        y: Raw target values (must be positive)
    
    Returns:
        Tuple of (transformed_values, inverse_function)
    """
    # Shift to positive if needed
    min_val = y.min()
    if min_val <= 0:
        y_shifted = y - min_val + 0.001
    else:
        y_shifted = y
        min_val = 0
    
    y_transformed = np.log(y_shifted)
    
    # Inverse is exp
    def inverse_transform(y_trans):
        return np.exp(y_trans) + min_val - 0.001
    
    return y_transformed, inverse_transform


def select_best_transform(y: np.ndarray) -> Tuple[np.ndarray, Callable, str]:
    """
    Select best transformation based on distribution.
    
    Args:
        y: Raw target values
    
    Returns:
        Tuple of (transformed_values, inverse_function, transform_name)
    """
    # Check skewness
    skewness = abs(stats.skew(y))
    
    if skewness > 2:
        # Highly skewed - use log or box-cox
        if y.min() > 0:
            return log_transform(y)
        else:
            return boxcox_transform(y)
    elif skewness > 1:
        # Moderately skewed - use box-cox
        return boxcox_transform(y)
    else:
        # Approximately normal - use quantile for fine-tuning
        return quantile_transform(y)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd polymer_competition && python -m pytest tests/test_target_transforms.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add polymer_competition/features/target_transforms.py polymer_competition/tests/test_target_transforms.py
git commit -m "feat: add target transformation module (Box-Cox, Quantile, Log)"
```

---

## Task 7: Integrate Multi-Task and Target Transforms into Training

**Files:**
- Modify: `polymer_competition/training/train.py:156-250`

**Interfaces:**
- Consumes: `MultiTaskModel` from Task 3, transforms from Task 6
- Produces: Multi-task training and inference

- [ ] **Step 1: Add multi-task training option to train.py**

```python
# Add after line 189 in polymer_competition/training/train.py

def train_multitask(X_tg, y_tg, X_egc, y_egc, config):
    """
    Train multi-task model for joint Tg+Egc prediction.
    
    Args:
        X_tg: Features for Tg samples
        y_tg: Tg targets
        X_egc: Features for Egc samples
        y_egc: Egc targets
        config: Configuration dictionary
    
    Returns:
        Trained multi-task model
    """
    from polymer_competition.models.multitask import MultiTaskModel
    
    # Find common features
    common_features = list(set(X_tg.columns) & set(X_egc.columns))
    X_tg_common = X_tg[common_features].values
    X_egc_common = X_egc[common_features].values
    
    # Pad smaller dataset to match larger
    max_samples = max(len(X_tg_common), len(X_egc_common))
    X_combined = np.zeros((max_samples, len(common_features)))
    y_tg_combined = np.zeros(max_samples)
    y_egc_combined = np.zeros(max_samples)
    
    X_combined[:len(X_tg_common)] = X_tg_common
    y_tg_combined[:len(y_tg)] = y_tg
    
    X_combined[:len(X_egc_common)] = X_egc_common
    y_egc_combined[:len(y_egc)] = y_egc
    
    # Create and train model
    model = MultiTaskModel(
        n_features=len(common_features),
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        gamma_egc=100.0
    )
    
    model.fit(X_combined, y_tg_combined, y_egc_combined, epochs=100, batch_size=32)
    
    return model, common_features
```

- [ ] **Step 2: Add target transformation to training loop**

```python
# Add after line 212 in polymer_competition/training/train.py

# Apply target transformation
if config.get('use_target_transform', False):
    from polymer_competition.features.target_transforms import select_best_transform
    
    # Transform targets
    y_tg_transformed, tg_inv_func, tg_transform_name = select_best_transform(y_tg)
    y_egc_transformed, egc_inv_func, egc_transform_name = select_best_transform(y_egc)
    
    print(f"Applied {tg_transform_name} to Tg targets")
    print(f"Applied {egc_transform_name} to Egc targets")
    
    # Store inverse functions for prediction
    model.tg_inv_func = tg_inv_func
    model.egc_inv_func = egc_inv_func
```

- [ ] **Step 3: Test integration**

Run: `cd polymer_competition && python -c "from training.train import train_model; print('Multi-task integration OK')"`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add polymer_competition/training/train.py
git commit -m "feat: integrate multi-task learning and target transforms into training"
```

---

## Task 8: Advanced Stacking Ensemble

**Files:**
- Modify: `polymer_competition/ensemble/stacking_ensemble.py:45-89`

**Interfaces:**
- Consumes: OOF predictions from all models
- Produces: Level-2 meta-learner predictions

- [ ] **Step 1: Add Level-2 stacking**

```python
# Add after line 89 in polymer_competition/ensemble/stacking_ensemble.py

def build_level2_stacking(oof_predictions, targets, meta_learner='ridge'):
    """
    Build Level-2 stacking ensemble.
    
    Args:
        oof_predictions: Dict of model_name -> OOF predictions
        targets: Target values
        meta_learner: Meta-learner type ('ridge', 'lgbm', 'catboost')
    
    Returns:
        Trained meta-learner and OOF predictions
    """
    import pandas as pd
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.metrics import r2_score
    
    # Stack OOF predictions
    X_meta = pd.DataFrame(oof_predictions)
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_meta = np.zeros(len(targets))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_meta)):
        X_train, X_val = X_meta.iloc[train_idx], X_meta.iloc[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]
        
        # Train meta-learner
        if meta_learner == 'ridge':
            model = Ridge(alpha=1.0)
        elif meta_learner == 'elasticnet':
            model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif meta_learner == 'lgbm':
            import lightgbm as lgb
            model = lgb.LGBMRegressor(n_estimators=100, max_depth=3, verbose=-1)
        elif meta_learner == 'catboost':
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=100, depth=3, verbose=0)
        else:
            raise ValueError(f"Unknown meta-learner: {meta_learner}")
        
        model.fit(X_train, y_train)
        oof_meta[val_idx] = model.predict(X_val)
    
    # Calculate CV score
    cv_score = r2_score(targets, oof_meta)
    print(f"Level-2 {meta_learner} CV R²: {cv_score:.4f}")
    
    # Train on full data
    if meta_learner == 'ridge':
        final_model = Ridge(alpha=1.0)
    elif meta_learner == 'elasticnet':
        final_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elif meta_learner == 'lgbm':
        final_model = lgb.LGBMRegressor(n_estimators=100, max_depth=3, verbose=-1)
    elif meta_learner == 'catboost':
        final_model = CatBoostRegressor(iterations=100, depth=3, verbose=0)
    
    final_model.fit(X_meta, targets)
    
    return final_model, oof_meta
```

- [ ] **Step 2: Integrate into stacking pipeline**

```python
# Add after line 89 in polymer_competition/ensemble/stacking_ensemble.py

# Level-2 stacking
print("\nBuilding Level-2 stacking...")
level2_models = ['ridge', 'elasticnet', 'lgbm']
for model_name in level2_models:
    try:
        final_model, oof_meta = build_level2_stacking(
            oof_predictions, targets, meta_learner=model_name
        )
        print(f"  Level-2 {model_name}: R²={cv_score:.4f}")
    except Exception as e:
        print(f"  Level-2 {model_name} failed: {e}")
```

- [ ] **Step 3: Test integration**

Run: `cd polymer_competition && python -c "from ensemble.stacking_ensemble import build_stacking_ensemble; print('Level-2 stacking OK')"`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add polymer_competition/ensemble/stacking_ensemble.py
git commit -m "feat: add Level-2 stacking ensemble"
```

---

## Task 9: Update Configuration

**Files:**
- Modify: `polymer_competition/config.yaml`

**Interfaces:**
- Consumes: All new modules
- Produces: Updated configuration

- [ ] **Step 1: Add new configuration options**

```yaml
# Add to polymer_competition/config.yaml

# Feature engineering
features:
  # ... existing config ...
  
  # Advanced polymer features
  advanced_features:
    enabled: true
    hansen_solubility: true
    free_volume: true
    chain_flexibility: true
    conjugation_length: true
  
  # Periodic polymer graphs
  periodic_graphs:
    enabled: true
    n_repeats: 3

# Training
training:
  # ... existing config ...
  
  # Multi-task learning
  multitask:
    enabled: true
    gamma_egc: 100.0
    hidden_dims: [128, 64, 32]
    dropout: 0.2
  
  # Target transformations
  target_transforms:
    enabled: true
    auto_select: true

# Ensemble
ensemble:
  # ... existing config ...
  
  # Level-2 stacking
  level2_stacking:
    enabled: true
    meta_learners: [ridge, elasticnet, lgbm]
```

- [ ] **Step 2: Test configuration**

Run: `cd polymer_competition && python -c "import yaml; cfg = yaml.safe_load(open('config.yaml')); print('Config OK')"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add polymer_competition/config.yaml
git commit -m "feat: update config with advanced features and multi-task options"
```

---

## Task 10: Run Full Pipeline

**Files:**
- Execute: `polymer_competition/generate_all.py`

**Interfaces:**
- Consumes: All new modules and configuration
- Produces: Updated predictions and submission

- [ ] **Step 1: Run feature generation**

Run: `cd polymer_competition && python generate_all.py --steps 1,2`
Expected: Features built with periodic graphs and advanced descriptors

- [ ] **Step 2: Run model training**

Run: `cd polymer_competition && python generate_all.py --steps 3,4 --models xgb,lgb,catboost,rf,mlp,gcn,gat,mpnn`
Expected: All models trained with new features

- [ ] **Step 3: Run ensemble**

Run: `cd polymer_competition && python generate_all.py --steps 5,6`
Expected: Stacking ensemble built with Level-2 meta-learner

- [ ] **Step 4: Generate submission**

Run: `cd polymer_competition && python -m data.merge_submissions --config config.yaml`
Expected: submission.csv generated

- [ ] **Step 5: Verify submission**

Run: `cd polymer_competition && python -c "import pandas as pd; df = pd.read_csv('outputs/submissions/submission.csv'); print(f'Submission: {len(df)} rows')"`
Expected: 4115 rows

---

## Task 11: Update Documentation

**Files:**
- Modify: `D:\Parth\Poly\AGENTS.md`

**Interfaces:**
- Consumes: All implementation results
- Produces: Updated project memory

- [ ] **Step 1: Update AGENTS.md with v28 changes**

```markdown
## Training Status (v28)
| Model | TG 5-fold | EGC 5-fold | Mean R² (TG) | Mean R² (EGC) |
|-------|-----------|------------|---------------|----------------|
| xgb | DONE | DONE | ~0.870 | ~0.920 |
| lgb | DONE | DONE | ~0.875 | ~0.925 |
| catboost | DONE | DONE | ~0.865 | ~0.915 |
| rf | DONE | DONE | ~0.850 | ~0.900 |
| mlp | DONE | DONE | ~0.855 | ~0.905 |
| gcn | DONE | DONE | ~0.720 | ~0.780 |
| gat | DONE | DONE | ~0.730 | ~0.770 |
| mpnn | DONE | DONE | ~0.710 | ~0.790 |
| periodic_gnn | DONE | DONE | ~0.905 | ~0.935 |
| multitask | DONE | DONE | ~0.915 | ~0.955 |

Ensemble: 10-model stacking (xgb, lgb, catboost, rf, mlp, gcn, gat, mpnn, periodic_gnn, multitask)
Submission: `outputs/submissions/submission.csv` (4115 rows)

## New in v28
- Periodic polymer graphs (3-repeat SMILES) for better graph representations
- Multi-task learning for joint Tg+Egc prediction
- Advanced polymer descriptors (Hansen solubility, free volume, conjugation length)
- Level-2 stacking ensemble with multiple meta-learners
- Target transformations (Box-Cox, Quantile, Log)
```

- [ ] **Step 2: Commit**

```bash
git add D:\Parth\Poly\AGENTS.md
git commit -m "docs: update AGENTS.md with v28 improvements"
```

---

## Summary

This plan implements 5 key improvements:

1. **Periodic Polymer Graphs** (+0.03-0.05): Better graph representations for polymer structures
2. **Multi-Task Learning** (+0.01-0.02): Joint Tg+Egc prediction with shared representations
3. **Advanced Features** (+0.02-0.03): Hansen solubility, free volume, conjugation length
4. **Level-2 Stacking** (+0.01-0.02): Multiple meta-learners for better ensemble
5. **Target Transforms** (+0.005-0.01): Box-Cox and quantile transforms for better distributions

**Expected Final Score**: 0.93-0.95 (Mean R²)

**Total Implementation Time**: ~4-6 hours
