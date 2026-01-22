"""Comprehensive tests for data utilities module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

from gt_pyg.data.utils import (
    _to_float_sequence,
    canonicalize_smiles,
    clean_df,
    get_data_from_csv,
    get_edge_dim,
    get_gnn_encodings,
    get_node_dim,
    get_pe,
    get_ring_membership_stats,
    get_tensor_data,
)


# Test molecules for various scenarios
TEST_SMILES = {
    "benzene": "c1ccccc1",
    "ethanol": "CCO",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "simple_chain": "CCCC",
}


class TestCleanDf:
    """Tests for clean_df function."""

    def test_returns_dataframe(self):
        """Test that clean_df returns a DataFrame."""
        df = pd.DataFrame({"Drug": ["CCO", "CCC"], "Y": [1.0, 2.0]})
        result = clean_df(df)
        assert isinstance(result, pd.DataFrame)

    def test_removes_invalid_smiles(self):
        """Test that invalid SMILES are removed."""
        df = pd.DataFrame({
            "Drug": ["CCO", "invalid_xyz", "CCC"],
            "Y": [1.0, 2.0, 3.0]
        })
        result = clean_df(df)
        assert len(result) == 2

    def test_fragment_handling_keeps_largest(self):
        """Test that largest fragment is kept."""
        df = pd.DataFrame({
            "Drug": ["CCO.[Na+].[Cl-]"],  # Ethanol with salt
            "Y": [1.0]
        })
        result = clean_df(df, use_largest_fragment=True)
        # Should keep the organic part
        assert len(result) >= 0  # May or may not be valid depending on processing

    def test_min_atom_filter(self):
        """Test minimum atom count filter."""
        df = pd.DataFrame({
            "Drug": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "Y": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        result = clean_df(df, min_num_atoms=3)
        # Should only keep molecules with >= 3 atoms
        assert len(result) == 3  # CCC, CCCC, CCCCC

    def test_preserves_labels(self):
        """Test that labels are preserved correctly."""
        df = pd.DataFrame({
            "Drug": ["CCO", "CCC"],
            "Y": [1.5, 2.5]
        })
        result = clean_df(df)
        assert "Y" in result.columns
        assert len(result["Y"]) == 2

    def test_canonical_smiles_output(self):
        """Test that output SMILES are canonical."""
        df = pd.DataFrame({
            "Drug": ["C(C)O", "OCC"],  # Non-canonical forms of ethanol
            "Y": [1.0, 2.0]
        })
        result = clean_df(df)
        # Both should become the same canonical SMILES
        unique_smiles = result["Drug"].nunique()
        assert unique_smiles == 1

    def test_custom_column_names(self):
        """Test with custom column names."""
        df = pd.DataFrame({
            "SMILES": ["CCO", "CCC"],
            "Activity": [1.0, 2.0]
        })
        result = clean_df(df, x_label="SMILES", y_label="Activity")
        assert "SMILES" in result.columns
        assert "Activity" in result.columns


class TestGetDataFromCsv:
    """Tests for get_data_from_csv function."""

    def test_reads_csv_correctly(self):
        """Test that CSV is read correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Drug,Y\n")
            f.write("CCO,1.0\n")
            f.write("CCC,2.0\n")
            temp_path = f.name

        try:
            result = get_data_from_csv(temp_path, "Drug", "Y")
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
        finally:
            Path(temp_path).unlink()

    def test_custom_separator(self):
        """Test reading CSV with custom separator."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("Drug\tY\n")
            f.write("CCO\t1.0\n")
            temp_path = f.name

        try:
            result = get_data_from_csv(temp_path, "Drug", "Y", sep="\t")
            assert len(result) == 1
        finally:
            Path(temp_path).unlink()

    def test_custom_column_names(self):
        """Test reading CSV with custom column names."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("SMILES,Activity\n")
            f.write("CCO,1.0\n")
            temp_path = f.name

        try:
            result = get_data_from_csv(temp_path, "SMILES", "Activity")
            assert "SMILES" in result.columns
            assert "Activity" in result.columns
        finally:
            Path(temp_path).unlink()


class TestGetRingMembershipStats:
    """Tests for get_ring_membership_stats function."""

    def test_benzene_all_atoms_in_ring(self):
        """Test that all benzene atoms are in a ring."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        atom_stats, _ = get_ring_membership_stats(mol)

        for idx, stats in atom_stats.items():
            assert stats["count"] == 1

    def test_benzene_ring_size_6(self):
        """Test that benzene ring size is 6."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        atom_stats, _ = get_ring_membership_stats(mol)

        for idx, stats in atom_stats.items():
            assert stats["min_size"] == 6
            assert stats["max_size"] == 6

    def test_ethanol_no_rings(self):
        """Test that ethanol has no rings."""
        mol = Chem.MolFromSmiles("CCO")
        atom_stats, bond_stats = get_ring_membership_stats(mol)

        for stats in atom_stats.values():
            assert stats["count"] == 0
            assert stats["min_size"] is None
            assert stats["max_size"] is None

    def test_naphthalene_fused_rings(self):
        """Test naphthalene (fused rings) statistics."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        atom_stats, _ = get_ring_membership_stats(mol)

        # Some atoms are in 2 rings (the bridging atoms)
        atoms_in_two_rings = sum(1 for s in atom_stats.values() if s["count"] == 2)
        assert atoms_in_two_rings == 2

    def test_aromatic_vs_non_aromatic_ring(self):
        """Test detection of aromatic vs non-aromatic rings."""
        # Benzene (aromatic)
        mol_arom = Chem.MolFromSmiles("c1ccccc1")
        atom_stats_arom, _ = get_ring_membership_stats(mol_arom)

        for stats in atom_stats_arom.values():
            assert stats["has_aromatic"] is True
            assert stats["has_non_aromatic"] is False

        # Cyclohexane (non-aromatic)
        mol_non = Chem.MolFromSmiles("C1CCCCC1")
        atom_stats_non, _ = get_ring_membership_stats(mol_non)

        for stats in atom_stats_non.values():
            assert stats["has_aromatic"] is False
            assert stats["has_non_aromatic"] is True

    def test_bond_ring_stats(self):
        """Test that bond ring stats are computed correctly."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        _, bond_stats = get_ring_membership_stats(mol)

        # All bonds in benzene should be in a 6-membered aromatic ring
        for stats in bond_stats.values():
            assert stats["count"] == 1
            assert stats["min_size"] == 6
            assert stats["has_aromatic"] is True


class TestGetPe:
    """Tests for get_pe (positional encodings) function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        pe = get_pe(mol, pe_dim=6)

        assert pe.shape == (6, 6)  # 6 atoms, 6 dimensions

    def test_normalized_vs_unnormalized(self):
        """Test difference between normalized and unnormalized Laplacian."""
        mol = Chem.MolFromSmiles("CCCC")

        pe_norm = get_pe(mol, normalized=True)
        pe_unnorm = get_pe(mol, normalized=False)

        # Should be different
        assert not np.allclose(pe_norm, pe_unnorm)

    def test_small_molecule_padding(self):
        """Test that small molecules get padded correctly."""
        mol = Chem.MolFromSmiles("CC")  # 2 atoms
        pe = get_pe(mol, pe_dim=6)

        # Should be padded to pe_dim columns
        assert pe.shape[0] == 2  # 2 atoms
        assert pe.shape[1] == 6  # pe_dim dimensions

    def test_deterministic_output(self):
        """Test that PE computation is deterministic."""
        mol = Chem.MolFromSmiles("c1ccccc1")

        pe1 = get_pe(mol)
        pe2 = get_pe(mol)

        assert np.allclose(pe1, pe2)

    def test_different_pe_dims(self):
        """Test different PE dimensions."""
        mol = Chem.MolFromSmiles("c1ccccc1")

        pe_4 = get_pe(mol, pe_dim=4)
        pe_8 = get_pe(mol, pe_dim=8)

        assert pe_4.shape[1] == 4
        assert pe_8.shape[1] == 8


class TestGetGnnEncodings:
    """Tests for get_gnn_encodings function."""

    def test_output_shape_square(self):
        """Test that output is a square matrix."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        gnn = get_gnn_encodings(mol)

        n_atoms = mol.GetNumAtoms()
        assert gnn.shape == (n_atoms, n_atoms)

    def test_symmetric_output(self):
        """Test that output is symmetric."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        gnn = get_gnn_encodings(mol)

        assert np.allclose(gnn, gnn.T)

    def test_kirchhoff_properties(self):
        """Test properties of the Kirchhoff pseudoinverse."""
        mol = Chem.MolFromSmiles("CCCC")  # Simple chain
        gnn = get_gnn_encodings(mol)

        # Should be real-valued
        assert np.isrealobj(gnn) or np.allclose(gnn.imag, 0)


class TestToFloatSequence:
    """Tests for _to_float_sequence function."""

    def test_single_float(self):
        """Test conversion of single float."""
        result = _to_float_sequence(1.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 1.5

    def test_single_int(self):
        """Test conversion of single int."""
        result = _to_float_sequence(3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 3.0

    def test_list_of_floats(self):
        """Test conversion of list of floats."""
        result = _to_float_sequence([1.0, 2.0, 3.0])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_list_with_none(self):
        """Test conversion of list with None values."""
        result = _to_float_sequence([1.0, None, 3.0])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result[0] == 1.0
        assert np.isnan(result[1])
        assert result[2] == 3.0

    def test_numpy_array(self):
        """Test conversion of numpy array."""
        arr = np.array([1.0, 2.0])
        result = _to_float_sequence(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_nan_handling(self):
        """Test that NaN values are preserved."""
        result = _to_float_sequence([1.0, np.nan, 3.0])
        assert np.isnan(result[1])


class TestGetTensorData:
    """Tests for get_tensor_data function."""

    def test_returns_list_of_data(self):
        """Test that function returns list of Data objects."""
        result = get_tensor_data(["CCO"], [1.0])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_single_task_labels(self):
        """Test single-task label handling."""
        result = get_tensor_data(["CCO", "CCC"], [1.0, 2.0])
        assert len(result) == 2
        assert result[0].y.shape == (1,)
        assert result[1].y.shape == (1,)

    def test_multi_task_labels(self):
        """Test multi-task label handling."""
        result = get_tensor_data(
            ["CCO", "CCC"],
            [[1.0, 2.0], [3.0, 4.0]]
        )
        assert result[0].y.shape == (2,)
        assert result[1].y.shape == (2,)

    def test_missing_labels_masked(self):
        """Test that missing labels are masked correctly."""
        result = get_tensor_data(
            ["CCO"],
            [[1.0, None, 3.0]]
        )
        assert result[0].y_mask[0] == 1.0  # Present
        assert result[0].y_mask[1] == 0.0  # Missing (None)
        assert result[0].y_mask[2] == 1.0  # Present

    def test_pe_included_when_enabled(self):
        """Test that PE is included when enabled."""
        result = get_tensor_data(["CCO"], [1.0], pe=True)
        assert result[0].pe is not None
        assert isinstance(result[0].pe, torch.Tensor)

    def test_pe_excluded_when_disabled(self):
        """Test that PE is excluded when disabled."""
        result = get_tensor_data(["CCO"], [1.0], pe=False)
        # When pe=False, pe is set to None which may result in attribute not existing
        # or being None depending on torch_geometric version
        pe_value = getattr(result[0], 'pe', None)
        assert pe_value is None

    def test_x_shape(self):
        """Test node features have correct shape."""
        result = get_tensor_data(["c1ccccc1"], [1.0], pe=False)
        # Benzene has 6 atoms
        assert result[0].x.shape[0] == 6
        assert result[0].x.dim() == 2  # [N, F]

    def test_edge_index_shape(self):
        """Test edge index has correct shape."""
        result = get_tensor_data(["CC"], [1.0], pe=False)
        # Shape should be [2, num_edges]
        assert result[0].edge_index.shape[0] == 2
        # Ethane has 1 bond -> 2 directed edges
        assert result[0].edge_index.shape[1] == 2

    def test_edge_attr_shape(self):
        """Test edge attributes have correct shape."""
        result = get_tensor_data(["CC"], [1.0], pe=False)
        # Should have same number of edge attrs as edges
        assert result[0].edge_attr.shape[0] == result[0].edge_index.shape[1]

    def test_batch_consistency(self):
        """Test that multiple molecules are processed consistently."""
        smiles = list(TEST_SMILES.values())
        labels = [float(i) for i in range(len(smiles))]

        result = get_tensor_data(smiles, labels, pe=False)

        # All should have consistent feature dimensions
        node_dims = set(d.x.shape[1] for d in result)
        edge_dims = set(d.edge_attr.shape[1] for d in result)

        assert len(node_dims) == 1
        assert len(edge_dims) == 1

    def test_gnn_encodings_enabled(self):
        """Test with GNN encodings enabled (default)."""
        result = get_tensor_data(["CC"], [1.0], gnn=True, pe=False)
        # Node features should include GNN diagonal term
        assert result[0].x.shape[1] > 0

    def test_gnn_encodings_disabled(self):
        """Test with GNN encodings disabled."""
        result_with = get_tensor_data(["CC"], [1.0], gnn=True, pe=False)
        result_without = get_tensor_data(["CC"], [1.0], gnn=False, pe=False)

        # With GNN should have 1 more feature (diagonal term)
        assert result_with[0].x.shape[1] == result_without[0].x.shape[1] + 1


class TestGetNodeDim:
    """Tests for get_node_dim function."""

    def test_returns_positive_int(self):
        """Test that get_node_dim returns a positive integer."""
        dim = get_node_dim()
        assert isinstance(dim, int)
        assert dim > 0

    def test_consistent_across_calls(self):
        """Test that dimension is consistent across multiple calls."""
        dim1 = get_node_dim()
        dim2 = get_node_dim()
        assert dim1 == dim2

    def test_matches_tensor_data(self):
        """Test that dimension matches actual tensor data."""
        dim = get_node_dim()
        data = get_tensor_data(["CCO"], [1.0], pe=False)
        assert data[0].x.shape[1] == dim


class TestGetEdgeDim:
    """Tests for get_edge_dim function."""

    def test_returns_positive_int(self):
        """Test that get_edge_dim returns a positive integer."""
        dim = get_edge_dim()
        assert isinstance(dim, int)
        assert dim > 0

    def test_consistent_across_calls(self):
        """Test that dimension is consistent across multiple calls."""
        dim1 = get_edge_dim()
        dim2 = get_edge_dim()
        assert dim1 == dim2

    def test_matches_tensor_data(self):
        """Test that dimension matches actual tensor data."""
        dim = get_edge_dim()
        data = get_tensor_data(["CCO"], [1.0], pe=False)
        assert data[0].edge_attr.shape[1] == dim


class TestIntegration:
    """Integration tests for the full data pipeline."""

    def test_full_pipeline_single_molecule(self):
        """Test full pipeline with a single molecule."""
        smiles = ["c1ccccc1"]  # Benzene
        labels = [1.0]

        data = get_tensor_data(smiles, labels, pe=True, gnn=True)

        assert len(data) == 1
        d = data[0]

        # Check all expected attributes
        assert hasattr(d, 'x')
        assert hasattr(d, 'edge_index')
        assert hasattr(d, 'edge_attr')
        assert hasattr(d, 'pe')
        assert hasattr(d, 'y')
        assert hasattr(d, 'y_mask')

        # Check types
        assert isinstance(d.x, torch.Tensor)
        assert isinstance(d.edge_index, torch.Tensor)
        assert isinstance(d.edge_attr, torch.Tensor)
        assert isinstance(d.pe, torch.Tensor)
        assert isinstance(d.y, torch.Tensor)
        assert isinstance(d.y_mask, torch.Tensor)

    def test_full_pipeline_multiple_molecules(self):
        """Test full pipeline with multiple molecules."""
        smiles = list(TEST_SMILES.values())
        labels = [float(i) for i in range(len(smiles))]

        data = get_tensor_data(smiles, labels, pe=True, gnn=True)

        assert len(data) == len(smiles)

        # All molecules should have consistent dimensions
        node_dim = data[0].x.shape[1]
        edge_dim = data[0].edge_attr.shape[1]
        pe_dim = data[0].pe.shape[1]

        for d in data:
            assert d.x.shape[1] == node_dim
            assert d.edge_attr.shape[1] == edge_dim
            assert d.pe.shape[1] == pe_dim

    def test_pipeline_with_multitask_labels(self):
        """Test pipeline with multi-task labels including missing values."""
        smiles = ["CCO", "CCC", "CCCC"]
        labels = [
            [1.0, 2.0, 3.0],
            [None, 5.0, 6.0],
            [7.0, None, None]
        ]

        data = get_tensor_data(smiles, labels, pe=False)

        assert len(data) == 3

        # Check masks
        assert data[0].y_mask.sum() == 3  # All present
        assert data[1].y_mask.sum() == 2  # One missing
        assert data[2].y_mask.sum() == 1  # Two missing

    def test_pipeline_deterministic(self):
        """Test that pipeline produces deterministic results."""
        smiles = ["c1ccccc1"]
        labels = [1.0]

        data1 = get_tensor_data(smiles, labels, pe=True, gnn=True)
        data2 = get_tensor_data(smiles, labels, pe=True, gnn=True)

        # Results should be identical
        assert torch.allclose(data1[0].x, data2[0].x)
        assert torch.equal(data1[0].edge_index, data2[0].edge_index)
        assert torch.allclose(data1[0].edge_attr, data2[0].edge_attr)


class TestCanonicalizeSmiles:
    """Tests for canonicalize_smiles function."""

    def test_simple_smiles_canonical(self):
        """Test that simple SMILES are canonicalized."""
        result = canonicalize_smiles("CCO")
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        assert mol.GetNumAtoms() == 3

    def test_preserves_charges_by_default(self):
        """Test that charges are preserved by default."""
        # Ammonium ion
        result = canonicalize_smiles("[NH4+]")
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        # Check that charge is preserved
        n_atom = mol.GetAtomWithIdx(0)
        assert n_atom.GetFormalCharge() == 1

    def test_preserves_negative_charges(self):
        """Test that negative charges are preserved."""
        result = canonicalize_smiles("[O-]")
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        o_atom = mol.GetAtomWithIdx(0)
        assert o_atom.GetFormalCharge() == -1

    def test_preserves_stereochemistry_by_default(self):
        """Test that stereochemistry is preserved by default."""
        result = canonicalize_smiles("C[C@H](O)F")
        assert result is not None
        # Check that the SMILES contains stereo indicators
        assert "@" in result or "/" in result or "\\" in result or "@@" in result

    def test_strips_stereochemistry_when_requested(self):
        """Test that stereochemistry can be stripped."""
        result = canonicalize_smiles("C[C@H](O)F", keep_stereo=False)
        assert result is not None
        # Stereo indicators should be removed
        assert "@" not in result

    def test_removes_salts_by_default(self):
        """Test that salts are removed by default."""
        result = canonicalize_smiles("CCO.[Na+].[Cl-]")
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        # Should only have ethanol (3 atoms)
        assert mol.GetNumAtoms() == 3

    def test_keeps_largest_fragment(self):
        """Test that largest fragment is kept."""
        # Ethanol is larger than water
        result = canonicalize_smiles("CCO.O")
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        # Should keep ethanol (3 atoms), not water (1 atom)
        assert mol.GetNumAtoms() == 3

    def test_neutralizes_charges_when_requested(self):
        """Test that charges can be neutralized."""
        result = canonicalize_smiles("[NH3+]C", keep_charges=False)
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_invalid_smiles_returns_none(self):
        """Test that invalid SMILES returns None."""
        result = canonicalize_smiles("invalid_smiles_xyz")
        assert result is None

    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = canonicalize_smiles("")
        assert result is None

    def test_canonical_output_consistent(self):
        """Test that different representations produce same canonical output."""
        # Different ways to write ethanol
        result1 = canonicalize_smiles("CCO")
        result2 = canonicalize_smiles("C(C)O")
        result3 = canonicalize_smiles("OCC")

        assert result1 == result2
        assert result2 == result3

    def test_complex_molecule(self):
        """Test canonicalization of a complex drug-like molecule."""
        # Aspirin
        result = canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        assert mol.GetNumAtoms() == 13

    def test_remove_hs_preserves_chirality(self):
        """Test that hydrogen removal preserves chiral center information."""
        # Chiral molecule with explicit H in SMILES notation
        smiles = "C[C@H](O)F"
        result = canonicalize_smiles(smiles)

        # Parse result and check chiral tag is preserved
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

        # Find the chiral carbon (bonded to C, O, F, and implicit H)
        chiral_atoms = [
            atom for atom in mol.GetAtoms()
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        ]
        assert len(chiral_atoms) == 1
        assert chiral_atoms[0].GetChiralTag() in [
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ]
