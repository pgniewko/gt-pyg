"""Tests for data/atom_features.py."""

import logging
import warnings

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from gt_pyg.data.atom_features import (
    PERMITTED_ATOMS,
    PERIOD_CATEGORIES,
    GROUP_CATEGORIES,
    RING_COUNT_CATEGORIES,
    RING_SIZE_CATEGORIES,
    get_atom_feature_dim,
    get_atom_features,
    get_gasteiger_charge,
    get_group,
    get_period,
    one_hot_encoding,
)


# ---------------------------------------------------------------------------
# one_hot_encoding
# ---------------------------------------------------------------------------

class TestOneHotEncoding:

    def test_known_value(self):
        result = one_hot_encoding("C", ["C", "N", "O", "Unknown"])
        assert result == [1, 0, 0, 0]

    def test_last_element(self):
        result = one_hot_encoding("O", ["C", "N", "O", "Unknown"])
        assert result == [0, 0, 1, 0]

    def test_unknown_maps_to_last(self):
        result = one_hot_encoding("Pu", PERMITTED_ATOMS)
        assert result[-1] == 1
        assert sum(result) == 1

    def test_unknown_logs_debug(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="gt_pyg.data.atom_features"):
            one_hot_encoding("Pu", PERMITTED_ATOMS)
        assert "Unknown value" in caplog.text

    def test_integer_encoding(self):
        result = one_hot_encoding(2, [0, 1, 2, 3])
        assert result == [0, 0, 1, 0]

    def test_length_matches_permitted_list(self):
        result = one_hot_encoding("C", PERMITTED_ATOMS)
        assert len(result) == len(PERMITTED_ATOMS)


# ---------------------------------------------------------------------------
# get_gasteiger_charge
# ---------------------------------------------------------------------------

class TestGasteigerCharge:

    def test_returns_float_in_range(self):
        mol = Chem.MolFromSmiles("CCO")
        rdPartialCharges.ComputeGasteigerCharges(mol)
        for atom in mol.GetAtoms():
            charge = get_gasteiger_charge(atom)
            assert -1.0 <= charge <= 1.0

    def test_missing_charge_returns_zero(self):
        mol = Chem.MolFromSmiles("C")
        # Don't compute charges — property is missing
        charge = get_gasteiger_charge(mol.GetAtomWithIdx(0))
        assert charge == 0.0

    def test_custom_clip(self):
        mol = Chem.MolFromSmiles("C")
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charge = get_gasteiger_charge(mol.GetAtomWithIdx(0), clip=1.0)
        assert -1.0 <= charge <= 1.0


# ---------------------------------------------------------------------------
# get_period / get_group
# ---------------------------------------------------------------------------

class TestPeriodAndGroup:

    @pytest.mark.parametrize("atomic_num,expected", [
        (1, 1), (6, 2), (11, 3), (26, 4), (53, 5), (79, 6), (87, 7),
    ])
    def test_period(self, atomic_num, expected):
        assert get_period(atomic_num) == expected

    def test_period_zero_for_invalid(self):
        assert get_period(0) == 0

    @pytest.mark.parametrize("atomic_num,expected", [
        (1, 1), (6, 14), (8, 16), (26, 8), (2, 18),
    ])
    def test_group(self, atomic_num, expected):
        assert get_group(atomic_num) == expected

    def test_group_lanthanide_returns_zero(self):
        # Europium (63) is a lanthanide
        assert get_group(63) == 0


# ---------------------------------------------------------------------------
# get_atom_features
# ---------------------------------------------------------------------------

class TestGetAtomFeatures:

    @pytest.fixture
    def ethanol_mol(self):
        mol = Chem.MolFromSmiles("CCO")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        rdPartialCharges.ComputeGasteigerCharges(mol)
        return mol

    def test_returns_1d_array(self, ethanol_mol):
        feat = get_atom_features(ethanol_mol.GetAtomWithIdx(0))
        assert isinstance(feat, np.ndarray)
        assert feat.ndim == 1

    def test_dimension_matches_get_atom_feature_dim(self, ethanol_mol):
        feat = get_atom_features(ethanol_mol.GetAtomWithIdx(0))
        assert len(feat) == get_atom_feature_dim()

    def test_no_stereochemistry_shorter(self, ethanol_mol):
        with_stereo = get_atom_features(ethanol_mol.GetAtomWithIdx(0), use_stereochemistry=True)
        no_stereo = get_atom_features(ethanol_mol.GetAtomWithIdx(0), use_stereochemistry=False)
        assert len(no_stereo) < len(with_stereo)

    def test_gnm_value_included(self, ethanol_mol):
        feat = get_atom_features(ethanol_mol.GetAtomWithIdx(0), gnm_value=0.42)
        # GNM is the last element
        assert feat[-1] == pytest.approx(0.42)

    def test_gnm_none_defaults_to_zero(self, ethanol_mol):
        feat = get_atom_features(ethanol_mol.GetAtomWithIdx(0), gnm_value=None)
        assert feat[-1] == 0.0


# ---------------------------------------------------------------------------
# get_atom_feature_dim
# ---------------------------------------------------------------------------

class TestGetAtomFeatureDim:

    def test_returns_positive_int(self):
        dim = get_atom_feature_dim()
        assert isinstance(dim, int)
        assert dim > 0

    def test_no_gasteiger_warning(self, caplog):
        """get_atom_feature_dim should not emit Gasteiger warnings (#13)."""
        with caplog.at_level(logging.WARNING):
            get_atom_feature_dim()
        gasteiger_msgs = [r for r in caplog.records if "gasteiger" in r.message.lower()]
        assert len(gasteiger_msgs) == 0

    def test_stereo_vs_no_stereo(self):
        with_stereo = get_atom_feature_dim(use_stereochemistry=True)
        no_stereo = get_atom_feature_dim(use_stereochemistry=False)
        assert with_stereo > no_stereo
