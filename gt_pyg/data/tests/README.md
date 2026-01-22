# Data Module Tests

Unit tests for the `gt_pyg.data` module covering atom featurization, bond featurization, and data utilities.

## Running Tests

```bash
# Run all data module tests
pytest gt_pyg/data/tests/ -v

# Run specific test file
pytest gt_pyg/data/tests/test_atom_features.py -v
pytest gt_pyg/data/tests/test_bond_features.py -v
pytest gt_pyg/data/tests/test_utils.py -v

# Run with coverage
pytest gt_pyg/data/tests/ --cov=gt_pyg.data --cov-report=term-missing
```

## Test Structure

- `test_atom_features.py` - Tests for atom/node featurization (50 tests)
- `test_bond_features.py` - Tests for bond/edge featurization (29 tests)
- `test_utils.py` - Tests for data preparation utilities (58 tests)

## Requirements

- pytest
- rdkit
- torch
- torch_geometric
- numpy
- pandas
