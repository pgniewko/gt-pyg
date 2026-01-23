"""gt_pyg: Graph Transformer Architecture in PyTorch Geometric.

This package provides an implementation of the Graph Transformer model
for molecular and graph-based machine learning tasks.

Modules:
    nn: Neural network components (GTConv, MLP, GraphTransformerNet)
    data: Data featurization and processing utilities
"""

from gt_pyg.nn import GTConv, MLP, GraphTransformerNet
from gt_pyg.data import get_tensor_data, canonicalize_smiles

__all__ = [
    "GTConv",
    "MLP",
    "GraphTransformerNet",
    "get_tensor_data",
    "canonicalize_smiles",
]
