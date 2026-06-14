# Chapter 13: Glossary

## Introduction

This glossary defines technical terms used in the PolyChain project. If you encounter an unfamiliar term, look it up here.

---

## A

### Aromatic
A chemical property where atoms are arranged in a ring with alternating single and double bonds. Examples: benzene ring (`c1ccccc1`). Aromatic compounds are often more stable and rigid.

### Atom-pair Fingerprint
A molecular fingerprint that encodes the topological distance between all pairs of atoms. Good at capturing molecular shape.

### Augmentation
Creating additional training data by modifying existing data. In PolyChain, SMILES randomization creates multiple equivalent representations of the same molecule.

### Average Pooling
A method to combine multiple values by taking their average. Used in GNNs to create graph-level embeddings from node embeddings.

---

## B

### Backpropagation
The algorithm used to train neural networks. It computes gradients of the loss with respect to model parameters, then updates parameters to reduce the loss.

### Backbone
The main feature extraction part of a neural network. In PolyChain, the GIN-S backbone processes molecular graphs.

### Batch Size
The number of samples processed together in one forward pass. Larger batches use more memory but may train faster.

### Batch Normalization
A technique that normalizes layer inputs to stabilize training. Helps neural networks train faster and more reliably.

### BERT-style Masking
A self-supervised learning technique where tokens are randomly masked and the model must predict them. Used in PolyChain's sub-SMILES masking task.

### Boosting
An ensemble method that trains models sequentially, with each model correcting errors of the previous one. XGBoost, LightGBM, and CatBoost are boosting algorithms.

---

## C

### Causal Mask
A mask that prevents tokens from attending to future tokens. In HAMF, scale k can only attend to scales ≤ k.

### Checkpoint
A saved snapshot of model parameters during training. Used to resume training or load the best model.

### ChemBERTa
A pre-trained language model for chemistry, trained on SMILES strings. Similar to BERT but for molecular text.

### Chemo-informatics
The application of computer science and information technology to chemistry. PolyChain uses RDKit for chemo-informatics tasks.

### Cosine Annealing
A learning rate schedule that smoothly decreases the learning rate following a cosine curve. Helps models converge to good solutions.

### Cross-Attention
A mechanism where one sequence attends to another. In HAMF, scale embeddings attend to each other across scales.

### Cross-Validation
A technique to evaluate model performance by splitting data into multiple folds, training on some and validating on others.

### CST (Chain Statistics Token)
A fixed-dimensional vector containing polymer-specific features computed from SMILES alone.

### CUDA
NVIDIA's parallel computing platform for GPU acceleration. PyTorch uses CUDA for fast neural network training.

---

## D

### Dimer
A molecule consisting of two repeat units. In PolyChain, the dimer graph is constructed by joining two copies of the monomer.

### Dropout
A regularization technique that randomly sets some neurons to zero during training. Prevents overfitting.

### EDA (Exploratory Data Analysis)
The process of analyzing data to find patterns, anomalies, and relationships before building models.

### Edge
A connection between two nodes in a graph. In molecular graphs, edges represent chemical bonds.

### Embedding
A learned representation of data in a continuous vector space. In PolyChain, atoms and graphs are embedded as vectors.

### Ensemble
A combination of multiple models to improve predictions. PolyChain ensembles 11 different models.

### Equivariant
A property where transformations of input produce corresponding transformations of output. PECGN is equivariant to SMILES translation.

---

## F

### Feature
An individual measurable property of data. In PolyChain, features include fingerprints, descriptors, and custom polymer features.

### Feature Matrix
A 2D array where rows are samples and columns are features. Shape: `(n_samples, n_features)`.

### Fingerprint
A binary vector representing molecular structure. Different fingerprint types capture different aspects of molecular structure.

### Fine-tuning
Training a pre-trained model on a specific task. PolyChain fine-tunes the backbone on property prediction.

### Forward Pass
The process of computing model output from input data. In PolyChain: SMILES → graphs → embeddings → prediction.

### Fusion
Combining information from multiple sources. HAMF fuses monomer, dimer, and trimer embeddings.

---

## G

### GCN (Graph Convolutional Network)
A neural network that operates on graphs by aggregating neighbor information. One of the baseline GNN models.

### GAT (Graph Attention Network)
A GNN that uses attention to weight neighbor contributions. More expressive than GCN.

### Gradient Clipping
Limiting gradient magnitude to prevent exploding gradients during training.

### Graph
A mathematical structure consisting of nodes (vertices) and edges. In PolyChain, molecules are represented as graphs.

### Graph-level Embedding
A single vector representing an entire graph. Created by pooling node embeddings.

### GroupKFold
A cross-validation strategy that ensures samples in the same group (e.g., same scaffold) stay together.

---

## H

### HAMF (Hierarchy-Aware Multi-Scale Fusion)
PolyChain's first innovation. Fuses monomer/dimer/trimer embeddings using chain-structured cross-attention.

### Hidden Dim
The dimensionality of hidden layers in a neural network. Controls model capacity.

### Hyperparameter
A parameter set before training (not learned from data). Examples: learning rate, batch size, number of layers.

---

## I

### Imputation
Filling missing values in data. Common strategies: mean, median, mode, or model-based imputation.

### Invariance
A property where transformations of input do not change the output. PECGN is invariant to SMILES translation.

---

## K

### KFold
A cross-validation strategy that splits data into K equal folds.

---

## L

### Label Smoothing
A regularization technique that softens hard labels (0/1) to (ε/1-ε). Prevents overconfident predictions.

### Layer Normalization
A technique that normalizes inputs across features (not batch). Stabilizes transformer training.

### Learning Rate
The step size for parameter updates during training. Too large: unstable; too small: slow convergence.

### LightGBM
A fast, efficient gradient boosting library. One of the tree-based models in PolyChain.

---

## M

### MACCS Keys
A set of 167 structural keys representing common molecular patterns. Used as a fingerprint type.

### Masked Language Modeling (MLM)
A self-supervised task where tokens are masked and the model predicts them. Used in PolyChain's sub-SMILES masking.

### Message Passing
The core operation in GNNs where nodes exchange information with neighbors.

### Morgan Fingerprint
A circular fingerprint that encodes atomic environments within a specified radius. Widely used in chemo-informatics.

### Molecule
A group of atoms bonded together. In PolyChain, molecules are represented as SMILES strings or graphs.

### Multi-head Attention
An attention mechanism that uses multiple attention heads to capture different types of relationships.

### Multi-scale
Operating at multiple levels of abstraction. PolyChain uses monomer, dimer, and trimer scales.

---

## N

### Node
A point in a graph. In molecular graphs, nodes represent atoms.

### Normalization
Scaling data to a standard range. In PolyChain, CST features are z-score normalized.

---

## O

### OOF (Out-of-Fold)
Predictions made on validation data during cross-validation. Used to evaluate model performance without data leakage.

### Overfitting
When a model learns noise in training data instead of underlying patterns. Prevented by regularization, early stopping, and cross-validation.

---

## P

### Parquet
A columnar data storage format. Efficient for large datasets. Used to store feature matrices.

### Permutation Invariance
A property where reordering nodes does not change the output. Inherited from GIN aggregator.

### Periodic
Repeating infinitely. Polymers are periodic chains. PECGN models this periodicity.

### Pooling
Combining multiple embeddings into a single embedding. Types: sum, mean, max, attention.

### Pre-training
Training a model on a large dataset before fine-tuning on a specific task. PolyChain uses self-supervised pre-training.

### PyTorch
A deep learning framework. PolyChain uses PyTorch and PyTorch Geometric.

### PyTorch Geometric (PyG)
A library for deep learning on graphs. Provides GNN layers, data loaders, and utilities.

---

## Q

### Quantile
A value below which a certain percentage of data falls. Used in EDA to check distribution shape.

---

## R

### Regression
A machine learning task where the goal is to predict a continuous value. PolyChain predicts properties like Tg and density.

### Regularization
Techniques to prevent overfitting. Examples: dropout, weight decay, label smoothing.

### Repeat Unit
The smallest repeating pattern in a polymer. Represented as SMILES with `*` connection points.

### Ridge Regression
Linear regression with L2 regularization. Prevents large coefficient values.

### RMSE (Root Mean Squared Error)
A metric measuring prediction error. Lower is better. Primary metric in polymer competitions.

---

## S

### Scaffold
A molecular substructure used for grouping. In PolyChain, RDKit Murcko scaffolds are used for cross-validation grouping.

### SMILES (Simplified Molecular Input Line Entry System)
A text representation of molecular structure. Example: `*CCO*` for polyethylene glycol repeat unit.

### SGD (Stochastic Gradient Descent)
An optimization algorithm that updates parameters using mini-batches of data.

### Span
The number of tokens in a sequence. Important for transformer models with fixed context windows.

### Split
Dividing data into training and validation sets. PolyChain uses GroupKFold splits.

### Substructure Matching
Finding if a specific pattern (SMARTS) exists in a molecule. Used to count end-groups.

---

## T

### Target
The variable we're trying to predict. In PolyChain: Tg, Tm, density, etc.

### Test Set
Data used for final evaluation. Does not contain target values.

### Trimer
A molecule consisting of three repeat units. In PolyChain, the trimer graph is constructed by joining three copies.

### Transformer
A neural network architecture based on self-attention. PolyChain uses transformers in HAMF.

### Tanimoto Similarity
A measure of similarity between two binary vectors. Used for fingerprint comparison.

---

## U

### Underfitting
When a model is too simple to capture underlying patterns. Indicated by high training error.

---

## V

### Validation Set
Data used to evaluate model during training. Not used for training itself.

### Virtual Node
A learnable parameter that mediates global information exchange in GIN-S. Updated at each message-passing layer.

---

## W

### Wasserstein Distance
A measure of distance between probability distributions. Used in EDA to check train/test drift.

### Weight Decay
L2 regularization that penalizes large weights. Prevents overfitting in neural networks.

### Weisfeiler-Leman Test
A graph isomorphism test. GIN is the most expressive GNN under this test.

---

## X

### XGBoost
A fast, scalable gradient boosting library. One of the tree-based models in PolyChain.

---

## Quick Reference

| Term | Definition |
|------|-----------|
| SMILES | Text representation of molecular structure |
| `*` | Connection point in polymer SMILES |
| Fingerprint | Binary vector representing molecular structure |
| GNN | Graph Neural Network |
| HAMF | Hierarchy-Aware Multi-Scale Fusion |
| PECGN | Periodic Equivariant Chain-Growth Network |
| CST | Chain Statistics Token |
| OOF | Out-of-Fold predictions |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of determination |
| Ensemble | Combining multiple models |

---

## Key Takeaways

- This glossary covers all technical terms used in the project
- Terms are organized alphabetically for easy lookup
- The Quick Reference table provides one-line definitions
- When in doubt, look up the term here before proceeding
