>Notice: This is research code that will not necessarily be maintained in the future.
>The code is under development so make sure you are using the most recent version.
>We welcome bug reports and PRs but make no guarantees about fixes or responses.

DESCRIPTION
==================================================
```gt_pyg``` is an implementation of the Graph Transformer Network in PyG

<p align="center"><img src="./assets/gt_v0.5.pdf" width="400"></p>


INSTALL
=======

Create the conda environment:
    
```
conda env create -f environment.yml
```

Clone and install the software:
```
git clone https://github.com/pgniewko/gt-pyg.git
pip install .
```

USAGE
=====

### Therapeutics Data Commons
Details and the leaderboard can be found on the TDC [website](https://tdcommons.ai/benchmark/admet_group/overview/)




## Implementation notes:
1. SMILES to tensor. I could use [from_smiles](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_smiles) method, but it doesn't allow flexibility in featurizationa and also requires creating multiple embeddgings and summing them, instead of using a single Linear layer.

2. For simplicity we don't create a separate DataSet object as we operate on the small datasets. So we pass a list of `Data` objects to `DataLoader`, as described in the [documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)



## References:
1. https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/        
