> Notice: This is research code that will not necessarily be maintained in the future.  
> The code is under development, so make sure you are using the most recent version.  
> We welcome bug reports and PRs but make no guarantees about fixes or responses.

## DESCRIPTION

`gt_pyg` is an implementation of the **Graph Transformer Architecture** in [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).

<p align="center"><img src="./assets/gt_v0.5.png" width="600"></p>

This sketch provides an overview of the Graph Transformer Architecture (Dwivedi & Bresson, 2021).  
In essence, the model implements a **dot-product self-attention network** with a `softmask` function,  
where `softmask` is a `softmax` applied only over the non-zero elements of the adjacency matrix `(A + I)`.

<p align="center"><img src="./assets/gated_gnn.png" width="600"></p>

This figure illustrates the **gating mechanism** used in the GT model (Chen et al., 2023, *Bioinformatics*).

---

## INSTALL

Clone the repository and install:
```bash
git clone https://github.com/pgniewko/gt-pyg.git
cd gt-pyg
pip install .
```

---

## USAGE

The following code snippet demonstrates how to test the installation of `gt-pyg` and use the `GTConv` layer:

```python
import torch
from torch_geometric.data import Data
from gt_pyg.nn.gt_conv import GTConv

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
from gt_pyg.data import get_tensor_data

dataset = pd.read_csv('solubility.csv')
tr_dataset = get_tensor_data(dataset['SMILES'].tolist(), dataset['logS'].tolist())
train_loader = DataLoader(tr_dataset, batch_size=256)
```

---

## Implementation Notes

This implementation integrates the core principles of the  original **Graph Transformer**.

### 1. A Generalization of Transformer Networks to Graphs

*Dwivedi & Bresson, 2021*
The Graph Transformer extends self-attention to structured graph inputs
by incorporating relational and topological context into attention
computation.
This implementation closely follows that design while leveraging PyTorch
Geometric's efficient message-passing API.

-   Uses PyG's numerically stable `softmax`, which applies the
    Log-Sum-Exp trick to avoid overflow.
-   Supports multiple aggregators (`"sum"`, `"mean"`, `"max"`, `"std"`,
    etc.) via `MultiAggregation` for richer expressiveness, similar to
    PNA.
-   Disables `qkv_bias` by default for stability and reproducibility.
-   Applies dropout to attention coefficients (shared `dropout` rate,
    not a separate parameter).

### 2. A Gated Graph Transformer for Protein Complex Structure Quality Assessment

*Chen et al., 2023, Bioinformatics*
Introduces gating mechanisms that modulate information flow through both
nodes and edges.
This implementation optionally includes the same gating logic
(`gate=True`) to enhance representational power.
To reproduce the original (ungated) Graph Transformer, set
`qkv_bias=True` and `gate=False`.

------------------------------------------------------------------------

### Implementation Foundations

-   Certain patterns and initialization schemes are adapted from
    [`TransformerConv`](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/transformer_conv.py).
-   The design remains fully compatible with PyG's `propagate` API,
    `Data`, and `Batch` abstractions.
-   For SMILES-to-graph conversion, use `gt_pyg.data.get_tensor_data`.
    **Do not** use `torch_geometric.utils.from_smiles` — it produces
    incompatible feature vectors.
-   Small datasets are handled as in-memory lists of `Data` objects,
    avoiding custom dataset definitions.

---

## Public API

### Model

| Symbol | Description |
|--------|-------------|
| `GraphTransformerNet` | Full model with variational readout (`mu` + `log_var` heads) |
| `GTConv` | Single Graph Transformer convolution layer |
| `MLP` | Multi-layer perceptron used in readout heads |
| `GraphTransformerNet.from_config(cfg)` | Construct a model from a config dict |

### Checkpointing & Utilities

| Symbol | Description |
|--------|-------------|
| `model.save_checkpoint(path)` | Save model, optimizer, and metadata |
| `model.load_checkpoint(path)` | Restore from checkpoint |
| `get_checkpoint_info(path)` | Read checkpoint metadata without loading weights |
| `model.freeze(component)` | Freeze parameters by component name |
| `model.unfreeze(component)` | Unfreeze parameters |
| `model.get_frozen_status()` | Dict of frozen/unfrozen components |

### Data

| Symbol | Description |
|--------|-------------|
| `get_tensor_data(smiles, labels)` | SMILES + labels to list of PyG `Data` objects |
| `get_atom_feature_dim()` | Dimensionality of the atom feature vector |
| `get_gnm_encodings(adj)` | Kirchhoff pseudoinverse diagonal (GNM) |
| `canonicalize_smiles(smi)` | Canonical SMILES string |

---

## Developers

### Installation (dev)

```bash
pip install -e ".[dev]"
```

### Running tests

```bash
pytest gt_pyg/ -v
```

---

## REFERENCES

1. [A Generalization of Transformer Networks to Graphs](https://arxiv.org/abs/2012.09699)  
2. [A Gated Graph Transformer for Protein Complex Structure Quality Assessment and Its Performance in CASP15](https://academic.oup.com/bioinformatics/article/39/Supplement_1/i308/7210460)

---

## COPYRIGHT NOTICE

Copyright (C) 2023-Present   
**Pawel Gniewek**  
Email: gniewko.pablo@gmail.com  
License: MIT  
All rights reserved.
