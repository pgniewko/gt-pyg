"""Comprehensive tests for atom featurization module."""

import numpy as np
import pytest
from rdkit import Chem

from gt_pyg.data.atom_features import (
    GROUP_CATEGORIES,
    PERIOD_CATEGORIES,
    PERMITTED_ATOMS,
    RING_COUNT_CATEGORIES,
    RING_SIZE_CATEGORIES,
    get_atom_feature_dim,
    get_atom_features,
    get_group,
    get_period,
    one_hot_encoding,
)
from gt_pyg.data.utils import get_ring_membership_stats


# Test molecules for various scenarios
TEST_MOLECULES = {
    "benzene": "c1ccccc1",  # Aromatic, symmetric
    "ethanol": "CCO",  # Simple alcohol
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",  # Drug-like
    "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # Heterocycles
    "charged": "[NH4+]",  # Charged atom
    "metal": "[Mg]",  # Metal
    "chiral": "C[C@H](O)F",  # Chiral center
    "naphthalene": "c1ccc2ccccc2c1",  # Fused rings
    "cyclohexane": "C1CCCCC1",  # Non-aromatic ring
    "acetylene": "C#C",  # Triple bond, sp hybridization
}


class TestOneHotEncoding:
    """Tests for one_hot_encoding function."""

    def test_known_value_returns_correct_encoding(self):
        """Test that a known value is encoded correctly."""
        result = one_hot_encoding("C", ["C", "N", "O"])
        assert result == [1, 0, 0]

    def test_middle_value_encoding(self):
        """Test encoding of a middle value."""
        result = one_hot_encoding("N", ["C", "N", "O"])
        assert result == [0, 1, 0]

    def test_last_value_encoding(self):
        """Test encoding of the last value."""
        result = one_hot_encoding("O", ["C", "N", "O"])
        assert result == [0, 0, 1]

    def test_unknown_value_maps_to_last_element(self):
        """Test that unknown values map to the last element."""
        result = one_hot_encoding("X", ["C", "N", "O", "Unknown"])
        assert result == [0, 0, 0, 1]

    def test_output_length_matches_permitted_list(self):
        """Test that output length matches the permitted list."""
        permitted = ["A", "B", "C", "D", "E"]
        result = one_hot_encoding("B", permitted)
        assert len(result) == len(permitted)

    def test_exactly_one_hot(self):
        """Test that exactly one element is 1 and all others are 0."""
        result = one_hot_encoding("N", ["C", "N", "O", "S"])
        assert sum(result) == 1
        assert result[1] == 1

    def test_integer_encoding(self):
        """Test one-hot encoding with integers."""
        result = one_hot_encoding(2, [0, 1, 2, 3])
        assert result == [0, 0, 1, 0]

    def test_unknown_integer_maps_to_last(self):
        """Test that unknown integers map to the last element."""
        result = one_hot_encoding(99, [0, 1, 2, "MoreThanTwo"])
        assert result == [0, 0, 0, 1]


class TestGetPeriod:
    """Tests for get_period function."""

    def test_hydrogen_is_period_1(self):
        """Test that hydrogen (atomic num 1) is in period 1."""
        assert get_period(1) == 1

    def test_carbon_is_period_2(self):
        """Test that carbon (atomic num 6) is in period 2."""
        assert get_period(6) == 2

    def test_sodium_is_period_3(self):
        """Test that sodium (atomic num 11) is in period 3."""
        assert get_period(11) == 3

    def test_iron_is_period_4(self):
        """Test that iron (atomic num 26) is in period 4."""
        assert get_period(26) == 4

    def test_iodine_is_period_5(self):
        """Test that iodine (atomic num 53) is in period 5."""
        assert get_period(53) == 5

    def test_gold_is_period_6(self):
        """Test that gold (atomic num 79) is in period 6."""
        assert get_period(79) == 6

    def test_uranium_is_period_7(self):
        """Test that uranium (atomic num 92) is in period 7."""
        assert get_period(92) == 7

    def test_period_returns_valid_range(self):
        """Test that period always returns a value in [1, 7]."""
        for atomic_num in [1, 6, 26, 79, 92, 118]:
            period = get_period(atomic_num)
            assert 1 <= period <= 7


class TestGetGroup:
    """Tests for get_group function."""

    def test_hydrogen_is_group_1(self):
        """Test that hydrogen is in group 1."""
        assert get_group(1) == 1

    def test_carbon_is_group_14(self):
        """Test that carbon is in group 14."""
        assert get_group(6) == 14

    def test_oxygen_is_group_16(self):
        """Test that oxygen is in group 16."""
        assert get_group(8) == 16

    def test_nitrogen_is_group_15(self):
        """Test that nitrogen is in group 15."""
        assert get_group(7) == 15

    def test_fluorine_is_group_17(self):
        """Test that fluorine is in group 17 (halogen)."""
        assert get_group(9) == 17

    def test_noble_gas_is_group_18(self):
        """Test that neon (noble gas) is in group 18."""
        assert get_group(10) == 18

    def test_transition_metal_iron(self):
        """Test iron (transition metal) group."""
        # Iron is in group 8
        assert get_group(26) == 8

    def test_group_returns_valid_range(self):
        """Test that group returns a value in GROUP_CATEGORIES."""
        for atomic_num in [1, 6, 8, 26, 79]:
            group = get_group(atomic_num)
            assert group in GROUP_CATEGORIES


class TestGetAtomFeatures:
    """Tests for get_atom_features function."""

    def test_output_is_numpy_array(self):
        """Test that output is a numpy array."""
        mol = Chem.MolFromSmiles("C")
        atom = mol.GetAtomWithIdx(0)
        result = get_atom_features(atom)
        assert isinstance(result, np.ndarray)

    def test_output_shape_consistent(self):
        """Test that output shape is consistent across different atoms."""
        mol = Chem.MolFromSmiles("CCNO")
        atom_ring_stats, _ = get_ring_membership_stats(mol)

        features_list = []
        for atom in mol.GetAtoms():
            features = get_atom_features(atom, atom_ring_stats=atom_ring_stats)
            features_list.append(len(features))

        # All atoms should have the same feature dimension
        assert len(set(features_list)) == 1

    def test_carbon_features(self):
        """Test features for a carbon atom."""
        mol = Chem.MolFromSmiles("C")
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)

        # First part is one-hot for atom type, "C" should be first
        assert features[0] == 1  # Carbon is first in PERMITTED_ATOMS
        assert isinstance(features, np.ndarray)

    def test_nitrogen_features(self):
        """Test features for a nitrogen atom."""
        mol = Chem.MolFromSmiles("N")
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)

        # Nitrogen is second in PERMITTED_ATOMS
        assert features[1] == 1

    def test_oxygen_features(self):
        """Test features for an oxygen atom."""
        mol = Chem.MolFromSmiles("O")
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)

        # Oxygen is third in PERMITTED_ATOMS
        assert features[2] == 1

    def test_aromatic_vs_aliphatic_carbon(self):
        """Test difference between aromatic and aliphatic carbon."""
        benzene = Chem.MolFromSmiles("c1ccccc1")
        ethane = Chem.MolFromSmiles("CC")

        aromatic_c = benzene.GetAtomWithIdx(0)
        aliphatic_c = ethane.GetAtomWithIdx(0)

        feat_aromatic = get_atom_features(aromatic_c)
        feat_aliphatic = get_atom_features(aliphatic_c)

        # Features should be different (aromaticity flag differs)
        assert not np.array_equal(feat_aromatic, feat_aliphatic)

    def test_sp_sp2_sp3_hybridization(self):
        """Test different hybridization states."""
        # SP3: methane carbon
        mol_sp3 = Chem.MolFromSmiles("C")
        atom_sp3 = mol_sp3.GetAtomWithIdx(0)

        # SP2: ethene carbon
        mol_sp2 = Chem.MolFromSmiles("C=C")
        atom_sp2 = mol_sp2.GetAtomWithIdx(0)

        # SP: acetylene carbon
        mol_sp = Chem.MolFromSmiles("C#C")
        atom_sp = mol_sp.GetAtomWithIdx(0)

        feat_sp3 = get_atom_features(atom_sp3)
        feat_sp2 = get_atom_features(atom_sp2)
        feat_sp = get_atom_features(atom_sp)

        # All should be different due to hybridization encoding
        assert not np.array_equal(feat_sp3, feat_sp2)
        assert not np.array_equal(feat_sp2, feat_sp)
        assert not np.array_equal(feat_sp3, feat_sp)

    def test_formal_charge_encoding(self):
        """Test encoding of formal charges."""
        # Ammonium (positive charge)
        mol_pos = Chem.MolFromSmiles("[NH4+]")
        atom_pos = mol_pos.GetAtomWithIdx(0)

        # Neutral ammonia
        mol_neutral = Chem.MolFromSmiles("N")
        atom_neutral = mol_neutral.GetAtomWithIdx(0)

        feat_pos = get_atom_features(atom_pos)
        feat_neutral = get_atom_features(atom_neutral)

        # Should be different due to charge
        assert not np.array_equal(feat_pos, feat_neutral)

    def test_chirality_encoding_when_enabled(self):
        """Test that chirality is encoded when enabled."""
        mol = Chem.MolFromSmiles("C[C@H](O)F")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Get the chiral carbon (index 1)
        chiral_atom = mol.GetAtomWithIdx(1)

        feat_with_chiral = get_atom_features(chiral_atom, use_chirality=True)
        feat_without_chiral = get_atom_features(chiral_atom, use_chirality=False)

        # With chirality should have more features
        assert len(feat_with_chiral) > len(feat_without_chiral)

    def test_chirality_disabled(self):
        """Test feature extraction with chirality disabled."""
        mol = Chem.MolFromSmiles("C[C@H](O)F")
        atom = mol.GetAtomWithIdx(1)

        features = get_atom_features(atom, use_chirality=False)
        # Should still return valid features
        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_hydrogens_implicit_vs_explicit(self):
        """Test difference between implicit and explicit hydrogen handling."""
        mol = Chem.MolFromSmiles("C")
        atom = mol.GetAtomWithIdx(0)

        feat_implicit = get_atom_features(atom, hydrogens_implicit=True)
        feat_explicit = get_atom_features(atom, hydrogens_implicit=False)

        # Different hydrogen handling produces different feature lengths
        assert len(feat_implicit) != len(feat_explicit) or not np.array_equal(feat_implicit, feat_explicit)

    def test_ring_membership_features(self):
        """Test that ring membership stats are encoded."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        atom_ring_stats, _ = get_ring_membership_stats(mol)

        ring_atom = mol.GetAtomWithIdx(0)

        feat_with_stats = get_atom_features(ring_atom, atom_ring_stats=atom_ring_stats)
        feat_without_stats = get_atom_features(ring_atom, atom_ring_stats=None)

        # Should be different when ring stats are provided
        assert not np.array_equal(feat_with_stats, feat_without_stats)

    def test_ring_features_without_stats(self):
        """Test that features are valid even without ring stats."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        atom = mol.GetAtomWithIdx(0)

        features = get_atom_features(atom, atom_ring_stats=None)

        # Should still be valid
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))

    def test_unknown_atom_maps_to_unknown(self):
        """Test that unknown/rare atoms map to Unknown category."""
        # Use a rare element not in PERMITTED_ATOMS
        mol = Chem.MolFromSmiles("[Rn]")  # Radon
        atom = mol.GetAtomWithIdx(0)

        features = get_atom_features(atom)

        # Unknown slot is at the last position of atom type encoding
        unknown_idx = len(PERMITTED_ATOMS) - 1
        # The unknown bit should be 1
        assert features[unknown_idx] == 1

    def test_feature_dimension_getter(self):
        """Test that get_atom_feature_dim returns correct dimension."""
        dim = get_atom_feature_dim()

        # Should be positive integer
        assert isinstance(dim, int)
        assert dim > 0

        # Should match actual feature length
        mol = Chem.MolFromSmiles("C")
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)
        assert len(features) == dim

    def test_deterministic_output(self):
        """Test that feature extraction is deterministic."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        atom = mol.GetAtomWithIdx(0)

        features1 = get_atom_features(atom)
        features2 = get_atom_features(atom)

        assert np.array_equal(features1, features2)

    def test_all_test_molecules(self):
        """Test feature extraction works for all test molecules."""
        for name, smiles in TEST_MOLECULES.items():
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Failed to parse {name}: {smiles}"

            atom_ring_stats, _ = get_ring_membership_stats(mol)

            for atom in mol.GetAtoms():
                features = get_atom_features(atom, atom_ring_stats=atom_ring_stats)
                assert isinstance(features, np.ndarray), f"Failed for {name}"
                assert len(features) > 0, f"Empty features for {name}"
                assert not np.any(np.isnan(features)), f"NaN in features for {name}"

    def test_degree_encoding(self):
        """Test that atom degree is correctly encoded."""
        # Methane carbon has degree 0 (no heavy neighbors)
        mol_methane = Chem.MolFromSmiles("C")
        methane_c = mol_methane.GetAtomWithIdx(0)

        # Ethane carbon has degree 1
        mol_ethane = Chem.MolFromSmiles("CC")
        ethane_c = mol_ethane.GetAtomWithIdx(0)

        # Propane middle carbon has degree 2
        mol_propane = Chem.MolFromSmiles("CCC")
        propane_c = mol_propane.GetAtomWithIdx(1)

        feat_0 = get_atom_features(methane_c)
        feat_1 = get_atom_features(ethane_c)
        feat_2 = get_atom_features(propane_c)

        # All should be different due to degree encoding
        assert not np.array_equal(feat_0, feat_1)
        assert not np.array_equal(feat_1, feat_2)

    def test_in_ring_flag(self):
        """Test that in-ring flag is correctly set."""
        # Benzene: all atoms in ring
        mol_ring = Chem.MolFromSmiles("c1ccccc1")
        ring_atom = mol_ring.GetAtomWithIdx(0)

        # Ethane: no ring
        mol_no_ring = Chem.MolFromSmiles("CC")
        no_ring_atom = mol_no_ring.GetAtomWithIdx(0)

        feat_ring = get_atom_features(ring_atom)
        feat_no_ring = get_atom_features(no_ring_atom)

        # Should be different
        assert not np.array_equal(feat_ring, feat_no_ring)

    def test_period_encoding_in_features(self):
        """Test that period encoding is included in features."""
        # Carbon (period 2) vs Sodium (period 3)
        mol_c = Chem.MolFromSmiles("C")
        mol_na = Chem.MolFromSmiles("[Na]")

        feat_c = get_atom_features(mol_c.GetAtomWithIdx(0))
        feat_na = get_atom_features(mol_na.GetAtomWithIdx(0))

        # Should be different due to different periods
        assert not np.array_equal(feat_c, feat_na)

    def test_group_encoding_in_features(self):
        """Test that group encoding is included in features."""
        # Carbon (group 14) vs Nitrogen (group 15)
        mol_c = Chem.MolFromSmiles("C")
        mol_n = Chem.MolFromSmiles("N")

        feat_c = get_atom_features(mol_c.GetAtomWithIdx(0))
        feat_n = get_atom_features(mol_n.GetAtomWithIdx(0))

        # Should be different due to different groups
        assert not np.array_equal(feat_c, feat_n)


class TestAtomFeatureConsistency:
    """Tests for consistency across the atom feature module."""

    def test_permitted_atoms_has_unknown(self):
        """Test that PERMITTED_ATOMS includes Unknown as last element."""
        assert PERMITTED_ATOMS[-1] == "Unknown"

    def test_ring_categories_cover_expected_values(self):
        """Test that ring categories cover expected ranges."""
        # Ring count: 0, 1, 2, 3, MoreThanThree
        assert 0 in RING_COUNT_CATEGORIES
        assert 1 in RING_COUNT_CATEGORIES
        assert "MoreThanThree" in RING_COUNT_CATEGORIES

        # Ring size: 3, 4, 5, 6, 7, 8, 9, 10, MoreThanTen
        assert 5 in RING_SIZE_CATEGORIES  # 5-membered rings common
        assert 6 in RING_SIZE_CATEGORIES  # 6-membered rings common
        assert "MoreThanTen" in RING_SIZE_CATEGORIES

    def test_period_categories_complete(self):
        """Test that period categories cover all periods."""
        assert PERIOD_CATEGORIES == [1, 2, 3, 4, 5, 6, 7]

    def test_group_categories_complete(self):
        """Test that group categories cover all groups."""
        expected = list(range(19))  # 0-18
        assert GROUP_CATEGORIES == expected

    def test_feature_dim_matches_with_all_options(self):
        """Test feature dimension with various option combinations."""
        mol = Chem.MolFromSmiles("C")
        atom = mol.GetAtomWithIdx(0)

        # Default options
        feat_default = get_atom_features(atom)
        dim_default = get_atom_feature_dim()
        assert len(feat_default) == dim_default

        # Without chirality
        feat_no_chiral = get_atom_features(atom, use_chirality=False)
        dim_no_chiral = get_atom_feature_dim(use_chirality=False)
        assert len(feat_no_chiral) == dim_no_chiral

        # Without implicit hydrogens
        feat_no_implicit = get_atom_features(atom, hydrogens_implicit=False)
        dim_no_implicit = get_atom_feature_dim(hydrogens_implicit=False)
        assert len(feat_no_implicit) == dim_no_implicit
