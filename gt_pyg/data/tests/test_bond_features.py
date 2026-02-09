"""Tests for data/bond_features.py."""

import numpy as np
import pytest
from rdkit import Chem

from gt_pyg.data.bond_features import get_bond_feature_dim, get_bond_features
from gt_pyg.data.utils import get_ring_membership_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mol_and_bond(smiles: str, bond_idx: int = 0):
    mol = Chem.MolFromSmiles(smiles)
    return mol, mol.GetBondWithIdx(bond_idx)


# ---------------------------------------------------------------------------
# get_bond_features
# ---------------------------------------------------------------------------

class TestGetBondFeatures:

    def test_returns_1d_array(self):
        _, bond = _mol_and_bond("CC")
        feat = get_bond_features(bond)
        assert isinstance(feat, np.ndarray)
        assert feat.ndim == 1

    def test_dimension_matches_get_bond_feature_dim(self):
        _, bond = _mol_and_bond("CC")
        assert len(get_bond_features(bond)) == get_bond_feature_dim()

    def test_single_bond(self):
        _, bond = _mol_and_bond("CC")
        feat = get_bond_features(bond)
        # Bond type one-hot: [SINGLE, DOUBLE, TRIPLE, AROMATIC, OTHER]
        assert feat[0] == 1  # SINGLE
        assert feat[1] == 0  # not DOUBLE

    def test_double_bond(self):
        # C=C in ethene
        _, bond = _mol_and_bond("C=C")
        feat = get_bond_features(bond)
        assert feat[0] == 0  # not SINGLE
        assert feat[1] == 1  # DOUBLE

    def test_triple_bond(self):
        _, bond = _mol_and_bond("C#C")
        feat = get_bond_features(bond)
        assert feat[2] == 1  # TRIPLE

    def test_aromatic_bond(self):
        # Benzene bonds are aromatic when kekulize=False (default)
        mol = Chem.MolFromSmiles("c1ccccc1")
        bond = mol.GetBondWithIdx(0)
        feat = get_bond_features(bond)
        assert feat[3] == 1  # AROMATIC

    def test_conjugated_flag(self):
        # 1,3-butadiene: C=CC=C — middle bond is conjugated
        mol = Chem.MolFromSmiles("C=CC=C")
        middle_bond = mol.GetBondWithIdx(1)  # C-C single bond between two C=C
        feat = get_bond_features(middle_bond)
        assert feat[5] == 1  # conjugated flag

    def test_in_ring_flag(self):
        mol = Chem.MolFromSmiles("C1CCC1")  # cyclobutane
        bond = mol.GetBondWithIdx(0)
        feat = get_bond_features(bond)
        assert feat[6] == 1  # in-ring flag

    def test_not_in_ring(self):
        _, bond = _mol_and_bond("CC")
        feat = get_bond_features(bond)
        assert feat[6] == 0  # not in ring


# ---------------------------------------------------------------------------
# Stereochemistry
# ---------------------------------------------------------------------------

class TestStereochemistry:

    def test_with_stereo_longer(self):
        _, bond = _mol_and_bond("CC")
        with_stereo = get_bond_features(bond, use_stereochemistry=True)
        no_stereo = get_bond_features(bond, use_stereochemistry=False)
        assert len(with_stereo) > len(no_stereo)

    def test_no_stereo_dim(self):
        _, bond = _mol_and_bond("CC")
        no_stereo = get_bond_features(bond, use_stereochemistry=False)
        assert len(no_stereo) == get_bond_feature_dim(use_stereochemistry=False)


# ---------------------------------------------------------------------------
# Ring stats
# ---------------------------------------------------------------------------

class TestRingStats:

    def test_ring_stats_populated(self):
        mol = Chem.MolFromSmiles("C1CCC1")  # cyclobutane
        _, bond_ring_stats = get_ring_membership_stats(mol)
        bond = mol.GetBondWithIdx(0)
        feat = get_bond_features(bond, bond_ring_stats=bond_ring_stats)
        assert len(feat) == get_bond_feature_dim()

    def test_ring_stats_none_gives_zeros(self):
        _, bond = _mol_and_bond("CC")
        feat = get_bond_features(bond, bond_ring_stats=None)
        # Last block: ring_count(5) + min_size(9) + max_size(9) + aromatic(1) + non_aromatic(1) = 25
        # All should be zero for no ring stats
        ring_block = feat[-25:]
        assert np.all(ring_block == 0)

    def test_benzene_aromatic_ring_flag(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        _, bond_ring_stats = get_ring_membership_stats(mol)
        bond = mol.GetBondWithIdx(0)
        feat = get_bond_features(bond, bond_ring_stats=bond_ring_stats)
        # Last two entries: in_any_aromatic_ring, in_any_non_aromatic_ring
        assert feat[-2] == 1  # aromatic ring
        assert feat[-1] == 0  # not non-aromatic

    def test_cyclohexane_non_aromatic_ring_flag(self):
        mol = Chem.MolFromSmiles("C1CCCCC1")
        _, bond_ring_stats = get_ring_membership_stats(mol)
        bond = mol.GetBondWithIdx(0)
        feat = get_bond_features(bond, bond_ring_stats=bond_ring_stats)
        assert feat[-2] == 0  # not aromatic
        assert feat[-1] == 1  # non-aromatic ring


# ---------------------------------------------------------------------------
# get_bond_feature_dim
# ---------------------------------------------------------------------------

class TestGetBondFeatureDim:

    def test_returns_positive_int(self):
        dim = get_bond_feature_dim()
        assert isinstance(dim, int)
        assert dim > 0

    def test_stereo_vs_no_stereo(self):
        with_stereo = get_bond_feature_dim(use_stereochemistry=True)
        no_stereo = get_bond_feature_dim(use_stereochemistry=False)
        assert with_stereo > no_stereo
