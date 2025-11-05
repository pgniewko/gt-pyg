> Notice: This is research code that will not necessarily be maintained in the future.  
> The code is under development, so make sure you are using the most recent version.  
> We welcome bug reports and PRs but make no guarantees about fixes or responses.

# DESCRIPTION

`gt_pyg` is an implementation of the **Graph Transformer Architecture** in [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).

<p align="center"><img src="./assets/gt_v0.5.png" width="600"></p>

This sketch provides an overview of the Graph Transformer Architecture (Dwivedi & Bresson, 2021).  
In essence, the model implements a **dot-product self-attention network** with a `softmask` function,  
where `softmask` is a `softmax` applied only over the non-zero elements of the adjacency matrix `(A + I)`.

<p align="center"><img src="./assets/gated_gnn.png" width="600"></p>

This figure illustrates the **gating mechanism** used in the GT model (Chen et al., 2023, *Bioinformatics*).

---

# INSTALL

Clone and install the software:
```bash
git clone https://github.com/pgniewko/gt-pyg.git
pip install .
```

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate gt
```

---

# USAGE

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
gt(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

The code also supports custom datasets. For example, if you have a file called `solubility.csv`  
with columns `SMILES` and `logS`, you can prepare a `DataLoader` object as follows:

```python
fn = 'solubility.csv'
x_label='SMILES'
y_label='logS'
dataset = get_data_from_csv(fn, x_label=x_label, y_label=y_label)
tr_dataset = get_tensor_data(dataset[x_label], dataset[y_label].to_list(), pe_dim=6)
train_loader = DataLoader(tr_dataset, batch_size=256)
```

---

# IMPLEMENTATION NOTES

This implementation integrates design principles from two key works:

1. **A Generalization of Transformer Networks to Graphs**  
   *(Dwivedi & Bresson, 2021)*  
   The Graph Transformer architecture extends self-attention to structured graph inputs,  
   using relational and topological context to modulate attention weights.  
   The current implementation replicates the original `GTConv` layer as closely as possible,  
   leveraging PyTorch Geometric’s efficient message passing API.  
   - The `softmask` function operates safely without clipping, as PyG’s `softmax` uses the Log-Sum-Exp trick.  
   - Aggregation defaults to `sum`, though multiple aggregators (as in PNA) can be used for richer expressiveness.  
   - Biases in the attention mechanism are set to `False` by default for stability.

2. **A Gated Graph Transformer for Protein Complex Structure Quality Assessment and its Performance in CASP15**  
   *(Chen et al., 2023, *Bioinformatics*)*  
   Introduces gating mechanisms that dynamically modulate attention flow.  
   This code includes the gating mechanism to improve the representational capacity of GT layers.  
   To reproduce the original GT model, set `qkv_bias=True` and `gate=False`.

Additional implementation notes:
- Some utility code is adapted from the [`TransformerConv`](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/transformer_conv.py) module.  
- For converting SMILES strings, consider using [`from_smiles`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_smiles).  
- For simplicity, small datasets are handled as lists of `Data` objects without defining a custom `Dataset` class.  
- The implementation follows a **Post-Normalization** Transformer setup for clarity and stability.

---

# REFERENCES

1. [A Generalization of Transformer Networks to Graphs](https://arxiv.org/abs/2012.09699)  
2. [A Gated Graph Transformer for Protein Complex Structure Quality Assessment and Its Performance in CASP15](https://academic.oup.com/bioinformatics/article/39/Supplement_1/i308/7210460)

---

# COPYRIGHT NOTICE

Copyright (C) 2023–  
**Pawel Gniewek**  
Email: gniewko.pablo@gmail.com  
License: MIT  
All rights reserved.
