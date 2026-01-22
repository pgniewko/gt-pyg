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

- `test_atom_features.py` - Tests for atom/node featurization
  - `TestOneHotEncoding` - One-hot encoding function
  - `TestGetPeriod` - Period (row) lookup
  - `TestGetGroup` - Group (column) lookup
  - `TestGetAtomFeatures` - Main atom feature extraction
  - `TestAtomFeatureConsistency` - Feature consistency checks
  - `TestPhysicochemicalFeatures` - Electronegativity, VdW, covalent radii
  - `TestGasteigerCharge` - Partial charge computation
  - `TestPharmacophoreFlags` - HBD, HBA, hydrophobic, ionizable detection
  - `TestNewAtomFeaturesIntegration` - Integration tests for new features

- `test_bond_features.py` - Tests for bond/edge featurization

- `test_utils.py` - Tests for data preparation utilities
  - `TestCleanSmilesOpenadmet` - Legacy SMILES cleaning (deprecated)
  - `TestCanonicalizeSmiles` - New SMILES canonicalization with charge/stereo preservation
  - `TestDeprecationWarning` - Deprecation warning for old function name
  - `TestCleanDf` - DataFrame cleaning
  - `TestGetDataFromCsv` - CSV loading
  - `TestGetRingMembershipStats` - Ring statistics
  - `TestGetPe` - Positional encodings
  - `TestGetGnnEncodings` - GNN-style encodings
  - `TestToFloatSequence` - Label conversion
  - `TestGetTensorData` - Full tensor data generation
  - `TestGetNodeDim` / `TestGetEdgeDim` - Dimension getters
  - `TestIntegration` - End-to-end pipeline tests

## Feature Dimensions

- **Node features**: 140 dimensions (131 original + 9 new)
  - Physicochemical: 3 (electronegativity, VdW radius, covalent radius)
  - Gasteiger charge: 1
  - Pharmacophore flags: 5 (HBD, HBA, hydrophobic, pos_ionizable, neg_ionizable)
- **Edge features**: 35 dimensions (unchanged)

## Requirements

- pytest
- rdkit
- torch
- torch_geometric
- numpy
- pandas
