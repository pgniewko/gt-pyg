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

To enable [ChEMBL structure pipeline](https://github.com/chembl/ChEMBL_Structure_Pipeline) standardization:
```bash
pip install -e ".[chembl]"
```

To install everything (dev + chembl + examples):
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

Notes:

- Pass `y=None` to `get_tensor_data(...)` for inference-only graphs.
- Multi-task labels may be sequences containing `None` or `np.nan`; `get_tensor_data(...)` adds a `y_mask` tensor so losses can ignore missing tasks.
- Use `ids=` to attach stable compound identifiers to warnings when a row is skipped (for example, because RDKit/Gasteiger charge computation failed).

---

## Public API

The tables below cover the primary supported entry points. For the exhaustive
current export lists, see [`gt_pyg/__init__.py`](gt_pyg/__init__.py),
[`gt_pyg/nn/__init__.py`](gt_pyg/nn/__init__.py), and
[`gt_pyg/data/__init__.py`](gt_pyg/data/__init__.py).

### Top-Level Import (`gt_pyg`)

| Symbol | Description |
|--------|-------------|
| `__version__` | Package version derived from the current git tag or installed metadata |
| `GraphTransformerNet`, `GTConv`, `MLP` | Core model components re-exported at the package top level |
| `get_tensor_data`, `get_atom_feature_dim`, `get_bond_feature_dim` | High-level data helpers re-exported at the package top level |

### Model (`gt_pyg.nn`)

| Symbol | Description |
|--------|-------------|
| `GraphTransformerNet` | Full graph-level model with variational readout (`mu` + `log_var` heads), configurable head depth (`num_head_layers`), optional head LayerNorm (`head_norm`), residual connections (`head_residual`), separate `head_dropout`, and optional latent return via `forward(..., return_latent=True)` |
| `GTConv` | Single Graph Transformer convolution layer |
| `MLP` | Multi-layer perceptron with optional LayerNorm and residual connections |
| `model.get_config()` / `GraphTransformerNet.from_config(config)` | Round-trip model configs for reproducible reconstruction |
| `model.num_parameters()` | Number of trainable parameters |

### Checkpointing & Utilities (`gt_pyg.nn`)

| Symbol | Description |
|--------|-------------|
| `model.save_checkpoint(path, ...)` | Save model weights, config, optional optimizer/scheduler state, and metadata |
| `GraphTransformerNet.load_checkpoint(path, ...)` | Reconstruct a new model from a checkpoint and return `(model, checkpoint_dict)` |
| `model.load_weights(path, ...)` | Load checkpoint weights into an existing model instance |
| `save_checkpoint(...)` / `load_checkpoint(...)` | Lower-level checkpoint helpers in [`gt_pyg/nn/checkpoint.py`](gt_pyg/nn/checkpoint.py) |
| `get_checkpoint_info(path)` | Read checkpoint metadata without loading weights |
| `model.freeze(components, exclude=None)` | Freeze parameters by component name (`embeddings`, `encoder`, `gt_layers`, `gt_layer_i`, `heads`, `pooling`, `all`) |
| `model.unfreeze(components)` | Unfreeze parameters |
| `model.get_frozen_status()` | Dict of frozen/unfrozen components |

### Data (`gt_pyg.data`)

| Symbol | Description |
|--------|-------------|
| `get_tensor_data(x_smiles, y=None, standardize=False, ids=None)` | Build PyG molecular `Data` objects from SMILES for single-task, multi-task, or inference-only use |
| `get_atom_features(atom, ...)` / `get_bond_features(bond, ...)` | Low-level featurizers for custom preprocessing pipelines |
| `get_atom_feature_dim(...)` / `get_bond_feature_dim(...)` | Dimensionality of the atom/bond feature vectors for the current featurization settings |
| `get_ring_membership_stats(mol)` / `get_pharmacophore_flags(mol)` | Reusable ring-statistics and pharmacophore preprocessing helpers |
| `get_gnm_encodings(adjacency)` | Kirchhoff pseudoinverse diagonal (GNM) |
| `canonicalize_smiles(...)` / `standardize_smiles(smiles)` | Canonicalization and optional ChEMBL structure-pipeline standardization |
| `PERMITTED_ATOMS`, `RING_COUNT_CATEGORIES`, `RING_SIZE_CATEGORIES`, `PERIOD_CATEGORIES`, `GROUP_CATEGORIES` | Exported feature vocabularies and category constants |

Detailed implementation notes live in
[`gt_pyg/nn/model.py`](gt_pyg/nn/model.py),
[`gt_pyg/data/utils.py`](gt_pyg/data/utils.py),
[`gt_pyg/data/atom_features.py`](gt_pyg/data/atom_features.py), and
[`gt_pyg/data/bond_features.py`](gt_pyg/data/bond_features.py).

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

## OpenADMET Benchmarks

The `examples/` directory contains training and evaluation notebooks for the
[OpenADMET](https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/) benchmark.

> Notebook compatibility: the notebooks below are pinned to the versions shown
> in their headings. If a notebook says `v1.6.0` or `v1.6.1b`, run it from that
> git tag; the current library API in this checkout may be newer.

### Single-task models (`v1.6.0`)

One `GraphTransformerNet` is trained per endpoint (a single model, not an
ensemble).  Training notebooks:

- [`train_logd.ipynb`](examples/train_logd.ipynb) — LogD
- [`train_ksol.ipynb`](examples/train_ksol.ipynb) — KSOL (solubility, trained on `LogS = log10((KSOL + 1) * 1e-6)`)

### Fine-tuned single-task models (`v1.6.1b`)

Same architecture as above, but initialized from a pretrained backbone
checkpoint (head weights are reinitialized).  The pretrained checkpoint is
produced by the [golem](https://github.com/pgniewko/golem) repository and
should be placed at `examples/pretrained/checkpoint.pt`.  Training notebooks:

- [`train_logd_finetune.ipynb`](examples/train_logd_finetune.ipynb) — LogD
- [`train_ksol_finetune.ipynb`](examples/train_ksol_finetune.ipynb) — KSOL (solubility)

### Ensemble model — beardy-polonium (`v1.3.0`)

The beardy-polonium submission is an **ensemble of 9 multi-task models** that
predicts all OpenADMET endpoints simultaneously.  The submission file is
stored in `examples/data/submissions/beardy-polonium-submission.csv`.

### Comparison

[`compare_predictions.ipynb`](examples/compare_predictions.ipynb) evaluates
the `v1.6.0` single-task models and `v1.6.1b` fine-tuned models against the
beardy-polonium ensemble on LogD and log-transformed KSOL. Metrics (MAE, RAE,
R², Spearman R, Kendall's Tau) are reported on the full test set, the
leaderboard (public) subset, and the private subset, with bootstrap
confidence intervals and significance tests.

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
