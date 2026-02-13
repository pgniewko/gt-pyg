> Notice: This is research code that will not necessarily be maintained in the future.
> The code is under development, so make sure you are using the most recent version.
> We welcome bug reports and PRs but make no guarantees about fixes or responses.

## DESCRIPTION

`gt_pyg` is an implementation of the **Graph Transformer Architecture** in [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).

---

## INSTALL

Clone the repository and install locally:
```bash
git clone https://github.com/pgniewko/gt-pyg.git
cd gt-pyg
python -m venv .venv
source .venv/bin/activate
pip install .
```

Alternatively, install directly from GitHub (call `source .venv/bin/activate` first):
```bash
pip install git+https://github.com/pgniewko/gt-pyg.git
```

To run the example notebooks, install with the `examples` extra:
```bash
pip install -e ".[examples]"
```

To install everything (dev + examples):
```bash
pip install -e ".[all]"
```

---

## USAGE

The following code snippet demonstrates how to test the installation of `gt-pyg` and use the `GTConv` layer:

```python
import torch
from gt_pyg import GTConv

num_nodes = 10
num_node_features = 3
num_edges = 20
num_edge_features = 2

# Generate random node features
x = torch.randn(num_nodes, num_node_features)

# Generate random edge indices
edge_index = torch.randint(high=num_nodes, size=(2, num_edges))

# Generate random edge attributes (optional)
edge_attr = torch.randn(num_edges, num_edge_features)

gt = GTConv(node_in_dim=num_node_features,
            edge_in_dim=num_edge_features,
            hidden_dim=15,
            num_heads=3)
x_out, edge_out = gt(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

The code also supports custom datasets. For example, if you have a file called `solubility.csv`
with columns `SMILES` and `logS`, you can prepare a `DataLoader` object as follows:

```python
import pandas as pd
from torch_geometric.loader import DataLoader
from gt_pyg import get_tensor_data

dataset = pd.read_csv('solubility.csv')
tr_dataset = get_tensor_data(dataset['SMILES'].tolist(), dataset['logS'].tolist())
train_loader = DataLoader(tr_dataset, batch_size=256)
```

---

## Public API

### Model (`gt_pyg.nn`)

| Symbol | Description |
|--------|-------------|
| `GraphTransformerNet` | Full model with variational readout (`mu` + `log_var` heads) |
| `GTConv` | Single Graph Transformer convolution layer |
| `MLP` | Multi-layer perceptron used in readout heads |
| `GraphTransformerNet.from_config(config)` | Construct a model from a config dict |

### Checkpointing & Utilities (`gt_pyg.nn`)

| Symbol | Description |
|--------|-------------|
| `model.save_checkpoint(path)` | Save model, optimizer, and metadata |
| `model.load_checkpoint(path)` | Restore from checkpoint |
| `get_checkpoint_info(path)` | Read checkpoint metadata without loading weights |
| `model.freeze(components)` | Freeze parameters by component name |
| `model.unfreeze(components)` | Unfreeze parameters |
| `model.get_frozen_status()` | Dict of frozen/unfrozen components |

### Data (`gt_pyg.data`)

| Symbol | Description |
|--------|-------------|
| `get_tensor_data(x_smiles, y)` | SMILES + labels to list of PyG `Data` objects |
| `get_atom_feature_dim()` | Dimensionality of the atom feature vector |
| `get_bond_feature_dim()` | Dimensionality of the bond feature vector |
| `get_gnm_encodings(adjacency)` | Kirchhoff pseudoinverse diagonal (GNM) |
| `canonicalize_smiles(smiles)` | Canonical SMILES string |

`GraphTransformerNet`, `GTConv`, `MLP`, and `get_tensor_data` are also available via the top-level `gt_pyg` import.

---

## Developers

### Installation (dev)

Clone the repository, create a virtual environment, and install in editable mode with dev dependencies:
```bash
git clone https://github.com/pgniewko/gt-pyg.git
cd gt-pyg
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running tests

```bash
pytest gt_pyg/ -v
```

---

## REFERENCES

1. [A Generalization of Transformer Networks to Graphs](https://arxiv.org/abs/2012.09699)
2. [A Gated Graph Transformer for Protein Complex Structure Quality Assessment
   (Chen et al., 2023, *Bioinformatics*)](https://academic.oup.com/bioinformatics/article/39/Supplement_1/i308/7210460)

---

## COPYRIGHT NOTICE

Copyright (C) 2023-Present          
**Pawel Gniewek**        
Email: gniewko.pablo@gmail.com          
License: MIT             
All rights reserved.         
