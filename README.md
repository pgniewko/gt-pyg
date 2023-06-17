# gt-pyg
Graph Transformer Network in PyG



## Implementation notes:
1. SMILES to tensor. I could use [from_smiles](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_smiles) method, but it doesn't allow flexibility in featurizationa and also requires creating multiple embeddgings and summing them, instead of using a single Linear layer.

2. For simplicity we don't create a separate DataSet object as we operate on the small datasets. So we pass a list of `Data` objects to `DataLoader`, as described in the [documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)



## References:
1. https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/        


