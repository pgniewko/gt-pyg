"""Comprehensive tests for bond featurization module."""

import numpy as np
import pytest
from rdkit import Chem

from gt_pyg.data.bond_features import (
    get_bond_feature_dim,
    get_bond_features,
)
from gt_pyg.data.utils import get_ring_membership_stats


# Test molecules for various bond scenarios
TEST_MOLECULES = {
    "ethane": "CC",  # Single bond
    "ethene": "C=C",  # Double bond
    "acetylene": "C#C",  # Triple bond
    "benzene": "c1ccccc1",  # Aromatic bonds
    "butadiene": "C=CC=C",  # Conjugated
    "propane": "CCC",  # Non-conjugated
    "cyclohexane": "C1CCCCC1",  # Ring, non-aromatic
    "trans_butene": r"C/C=C/C",  # E stereochemistry
    "cis_butene": r"C/C=C\C",  # Z stereochemistry
    "naphthalene": "c1ccc2ccccc2c1",  # Fused aromatic
}


class TestGetBondFeatures:
    """Tests for get_bond_features function."""

    def test_output_is_numpy_array(self):
        """Test that output is a numpy array."""
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)
        result = get_bond_features(bond)
        assert isinstance(result, np.ndarray)

    def test_output_shape_consistent(self):
        """Test that output shape is consistent across different bonds."""
        mol = Chem.MolFromSmiles("CCCC")
        _, bond_ring_stats = get_ring_membership_stats(mol)

        features_list = []
        for bond in mol.GetBonds():
            features = get_bond_features(bond, bond_ring_stats=bond_ring_stats)
            features_list.append(len(features))

        # All bonds should have the same feature dimension
        assert len(set(features_list)) == 1

    def test_single_bond_encoding(self):
        """Test features for a single bond."""
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # First element should be 1 (single bond is first in permitted list)
        assert features[0] == 1  # SINGLE bond
        assert features[1] == 0  # Not DOUBLE
        assert features[2] == 0  # Not TRIPLE
        assert features[3] == 0  # Not AROMATIC

    def test_double_bond_encoding(self):
        """Test features for a double bond."""
        mol = Chem.MolFromSmiles("C=C")
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Second element should be 1 (double bond)
        assert features[0] == 0  # Not SINGLE
        assert features[1] == 1  # DOUBLE bond
        assert features[2] == 0  # Not TRIPLE
        assert features[3] == 0  # Not AROMATIC

    def test_triple_bond_encoding(self):
        """Test features for a triple bond."""
        mol = Chem.MolFromSmiles("C#C")
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Third element should be 1 (triple bond)
        assert features[0] == 0  # Not SINGLE
        assert features[1] == 0  # Not DOUBLE
        assert features[2] == 1  # TRIPLE bond
        assert features[3] == 0  # Not AROMATIC

    def test_aromatic_bond_encoding(self):
        """Test features for an aromatic bond."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Fourth element should be 1 (aromatic bond)
        assert features[0] == 0  # Not SINGLE
        assert features[1] == 0  # Not DOUBLE
        assert features[2] == 0  # Not TRIPLE
        assert features[3] == 1  # AROMATIC bond

    def test_conjugated_bond(self):
        """Test features for a conjugated bond."""
        mol = Chem.MolFromSmiles("C=CC=C")  # Butadiene
        # The central single bond should be conjugated
        central_bond = mol.GetBondWithIdx(1)
        features = get_bond_features(central_bond)

        # Check conjugation flag (index 4 after bond type encoding)
        # Bond type (4) + conjugation (1)
        assert features[4] == 1  # Conjugated

    def test_non_conjugated_bond(self):
        """Test features for a non-conjugated bond."""
        mol = Chem.MolFromSmiles("CCC")  # Propane
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Check conjugation flag
        assert features[4] == 0  # Not conjugated

    def test_ring_bond(self):
        """Test features for a bond in a ring."""
        mol = Chem.MolFromSmiles("C1CCCCC1")  # Cyclohexane
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Check in-ring flag (index 5 after bond type + conjugation)
        assert features[5] == 1  # In ring

    def test_non_ring_bond(self):
        """Test features for a bond not in a ring."""
        mol = Chem.MolFromSmiles("CC")  # Ethane
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Check in-ring flag
        assert features[5] == 0  # Not in ring

    def test_stereochemistry_E(self):
        """Test features for E (trans) stereochemistry."""
        mol = Chem.MolFromSmiles(r"C/C=C/C")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Get the double bond
        double_bond = mol.GetBondWithIdx(1)
        features = get_bond_features(double_bond, use_stereochemistry=True)

        # Features should include stereo encoding
        assert len(features) > 6

    def test_stereochemistry_Z(self):
        """Test features for Z (cis) stereochemistry."""
        mol = Chem.MolFromSmiles(r"C/C=C\C")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Get the double bond
        double_bond = mol.GetBondWithIdx(1)
        features = get_bond_features(double_bond, use_stereochemistry=True)

        # Features should include stereo encoding
        assert len(features) > 6

    def test_stereochemistry_disabled(self):
        """Test feature extraction with stereochemistry disabled."""
        mol = Chem.MolFromSmiles(r"C/C=C/C")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        double_bond = mol.GetBondWithIdx(1)

        feat_with = get_bond_features(double_bond, use_stereochemistry=True)
        feat_without = get_bond_features(double_bond, use_stereochemistry=False)

        # Without stereochemistry should have fewer features
        assert len(feat_without) < len(feat_with)

    def test_ring_membership_features(self):
        """Test that ring membership stats are encoded."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        _, bond_ring_stats = get_ring_membership_stats(mol)

        bond = mol.GetBondWithIdx(0)

        feat_with_stats = get_bond_features(bond, bond_ring_stats=bond_ring_stats)
        feat_without_stats = get_bond_features(bond, bond_ring_stats=None)

        # Should be different when ring stats are provided
        assert not np.array_equal(feat_with_stats, feat_without_stats)

    def test_ring_features_without_stats(self):
        """Test that features are valid even without ring stats."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        bond = mol.GetBondWithIdx(0)

        features = get_bond_features(bond, bond_ring_stats=None)

        # Should still be valid
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))

    def test_feature_dimension_getter(self):
        """Test that get_bond_feature_dim returns correct dimension."""
        dim = get_bond_feature_dim()

        # Should be positive integer
        assert isinstance(dim, int)
        assert dim > 0

        # Should match actual feature length
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)
        assert len(features) == dim

    def test_feature_dimension_without_stereo(self):
        """Test feature dimension without stereochemistry."""
        dim_with = get_bond_feature_dim(use_stereochemistry=True)
        dim_without = get_bond_feature_dim(use_stereochemistry=False)

        assert dim_without < dim_with

        # Verify with actual features
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond, use_stereochemistry=False)
        assert len(features) == dim_without

    def test_deterministic_output(self):
        """Test that feature extraction is deterministic."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        bond = mol.GetBondWithIdx(0)

        features1 = get_bond_features(bond)
        features2 = get_bond_features(bond)

        assert np.array_equal(features1, features2)

    def test_all_bonds_in_molecule(self):
        """Test feature extraction for all bonds in a molecule."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        _, bond_ring_stats = get_ring_membership_stats(mol)

        feature_lengths = set()
        for bond in mol.GetBonds():
            features = get_bond_features(bond, bond_ring_stats=bond_ring_stats)
            assert isinstance(features, np.ndarray)
            assert not np.any(np.isnan(features))
            feature_lengths.add(len(features))

        # All bonds should have same feature dimension
        assert len(feature_lengths) == 1

    def test_all_test_molecules(self):
        """Test feature extraction works for all test molecules."""
        for name, smiles in TEST_MOLECULES.items():
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Failed to parse {name}: {smiles}"

            _, bond_ring_stats = get_ring_membership_stats(mol)

            for bond in mol.GetBonds():
                features = get_bond_features(bond, bond_ring_stats=bond_ring_stats)
                assert isinstance(features, np.ndarray), f"Failed for {name}"
                assert len(features) > 0, f"Empty features for {name}"
                assert not np.any(np.isnan(features)), f"NaN in features for {name}"


class TestBondFeatureComparisons:
    """Tests comparing bond features across different bond types."""

    def test_single_vs_double_vs_triple(self):
        """Test that single, double, and triple bonds have different features."""
        mol_single = Chem.MolFromSmiles("CC")
        mol_double = Chem.MolFromSmiles("C=C")
        mol_triple = Chem.MolFromSmiles("C#C")

        feat_single = get_bond_features(mol_single.GetBondWithIdx(0))
        feat_double = get_bond_features(mol_double.GetBondWithIdx(0))
        feat_triple = get_bond_features(mol_triple.GetBondWithIdx(0))

        assert not np.array_equal(feat_single, feat_double)
        assert not np.array_equal(feat_double, feat_triple)
        assert not np.array_equal(feat_single, feat_triple)

    def test_aromatic_vs_non_aromatic(self):
        """Test that aromatic and non-aromatic bonds are different."""
        mol_aromatic = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        mol_non_aromatic = Chem.MolFromSmiles("C1CCCCC1")  # Cyclohexane

        feat_aromatic = get_bond_features(mol_aromatic.GetBondWithIdx(0))
        feat_non_aromatic = get_bond_features(mol_non_aromatic.GetBondWithIdx(0))

        assert not np.array_equal(feat_aromatic, feat_non_aromatic)

    def test_ring_vs_non_ring_single_bond(self):
        """Test that ring and non-ring single bonds are different."""
        mol_ring = Chem.MolFromSmiles("C1CCCCC1")  # Cyclohexane
        mol_non_ring = Chem.MolFromSmiles("CCCCCC")  # Hexane

        feat_ring = get_bond_features(mol_ring.GetBondWithIdx(0))
        feat_non_ring = get_bond_features(mol_non_ring.GetBondWithIdx(0))

        assert not np.array_equal(feat_ring, feat_non_ring)


class TestBondRingStats:
    """Tests for bond ring statistics encoding."""

    def test_benzene_bond_ring_stats(self):
        """Test ring stats for benzene bonds."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        _, bond_ring_stats = get_ring_membership_stats(mol)

        # All bonds in benzene should be in exactly one 6-membered ring
        for idx, stats in bond_ring_stats.items():
            assert stats["count"] == 1
            assert stats["min_size"] == 6
            assert stats["max_size"] == 6
            assert stats["has_aromatic"] is True
            assert stats["has_non_aromatic"] is False

    def test_naphthalene_fused_ring_bonds(self):
        """Test ring stats for naphthalene (fused rings)."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        _, bond_ring_stats = get_ring_membership_stats(mol)

        # Find bonds that are in 2 rings (the shared bond)
        shared_bond_count = sum(1 for stats in bond_ring_stats.values() if stats["count"] == 2)
        # Naphthalene has 1 shared bond between the two 6-membered rings
        assert shared_bond_count == 1

    def test_cyclohexane_bond_stats(self):
        """Test ring stats for cyclohexane (non-aromatic ring)."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        _, bond_ring_stats = get_ring_membership_stats(mol)

        for idx, stats in bond_ring_stats.items():
            assert stats["count"] == 1
            assert stats["min_size"] == 6
            assert stats["max_size"] == 6
            assert stats["has_aromatic"] is False
            assert stats["has_non_aromatic"] is True

    def test_non_ring_bond_stats(self):
        """Test ring stats for bonds not in rings."""
        mol = Chem.MolFromSmiles("CC")
        _, bond_ring_stats = get_ring_membership_stats(mol)

        stats = bond_ring_stats[0]
        assert stats["count"] == 0
        assert stats["min_size"] is None
        assert stats["max_size"] is None
        assert stats["has_aromatic"] is False
        assert stats["has_non_aromatic"] is False
