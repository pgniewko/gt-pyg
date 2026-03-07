"""Tests for ChEMBL structure pipeline standardization."""

import unittest.mock

import pytest

from gt_pyg.data.utils import standardize_smiles, get_tensor_data


# ---------------------------------------------------------------------------
# Check that chembl_structure_pipeline is available for these tests
# ---------------------------------------------------------------------------

chembl_available = True
try:
    import chembl_structure_pipeline  # noqa: F401
except ImportError:
    chembl_available = False

needs_chembl = pytest.mark.skipif(
    not chembl_available,
    reason="chembl_structure_pipeline not installed",
)


# ---------------------------------------------------------------------------
# standardize_smiles
# ---------------------------------------------------------------------------

class TestStandardizeSmiles:

    @needs_chembl
    def test_simple_passthrough(self):
        result = standardize_smiles("CCO")
        assert result is not None
        assert result == "CCO"

    @needs_chembl
    def test_salt_stripping(self):
        result = standardize_smiles("c1ccccc1.Cl")
        assert result is not None
        # Parent should be benzene without HCl
        assert "Cl" not in result
        assert "c1ccccc1" == result

    @needs_chembl
    def test_returns_none_on_invalid(self):
        result = standardize_smiles("not_a_smiles")
        assert result is None

    @needs_chembl
    def test_returns_canonical(self):
        result = standardize_smiles("C(O)C")
        assert result is not None
        assert result == "CCO"

    def test_raises_without_chembl(self):
        with unittest.mock.patch.dict("sys.modules", {"chembl_structure_pipeline": None}):
            with pytest.raises(ImportError, match="chembl_structure_pipeline"):
                standardize_smiles("CCO")


# ---------------------------------------------------------------------------
# get_tensor_data with standardize flag
# ---------------------------------------------------------------------------

class TestGetTensorDataStandardize:

    @needs_chembl
    def test_standardize_flag_produces_valid_data(self):
        data_list = get_tensor_data(["CCO"], [1.0], standardize=True)
        assert len(data_list) == 1
        assert data_list[0].x is not None

    @needs_chembl
    def test_standardize_strips_salt(self):
        # With standardize, salt is stripped before featurization
        data_std = get_tensor_data(["c1ccccc1.Cl"], standardize=True)
        data_plain = get_tensor_data(["c1ccccc1"], standardize=False)
        # After salt stripping, atom count should match plain benzene
        assert data_std[0].x.shape[0] == data_plain[0].x.shape[0]

    def test_standardize_raises_without_chembl(self):
        with unittest.mock.patch.dict("sys.modules", {"chembl_structure_pipeline": None}):
            with pytest.raises(ImportError, match="chembl_structure_pipeline"):
                get_tensor_data(["CCO"], standardize=True)

    def test_standardize_false_no_import_needed(self):
        """standardize=False should work even without chembl installed."""
        data_list = get_tensor_data(["CCO"], [1.0], standardize=False)
        assert len(data_list) == 1
