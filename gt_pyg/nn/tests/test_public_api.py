"""Public API contract tests for package-level exports."""

import pytest

import gt_pyg
import gt_pyg.data
import gt_pyg.nn


README_TOP_LEVEL_EXPORTS = {
    "__version__",
    "GraphTransformerNet",
    "GTConv",
    "MLP",
    "get_tensor_data",
    "get_atom_feature_dim",
    "get_bond_feature_dim",
}

PUBLIC_MODULES = (gt_pyg, gt_pyg.data, gt_pyg.nn)


def test_readme_documented_top_level_symbols_importable():
    """README-documented top-level symbols should import from gt_pyg."""
    from gt_pyg import (  # noqa: PLC0415
        GTConv,
        GraphTransformerNet,
        MLP,
        __version__,
        get_atom_feature_dim,
        get_bond_feature_dim,
        get_tensor_data,
    )

    assert __version__
    assert GraphTransformerNet is gt_pyg.GraphTransformerNet
    assert GTConv is gt_pyg.GTConv
    assert MLP is gt_pyg.MLP
    assert get_tensor_data is gt_pyg.get_tensor_data
    assert get_atom_feature_dim is gt_pyg.get_atom_feature_dim
    assert get_bond_feature_dim is gt_pyg.get_bond_feature_dim


def test_top_level_exports_include_readme_documented_symbols():
    """README-documented top-level symbols should stay in gt_pyg.__all__."""
    assert README_TOP_LEVEL_EXPORTS <= set(gt_pyg.__all__)


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_public_exports_do_not_include_private_names(module):
    """Public package exports should not promote private helpers."""
    assert all(name == "__version__" or not name.startswith("_") for name in module.__all__)


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_public_exports_are_accessible_from_modules(module):
    """Every public export should be accessible from its package module."""
    for name in module.__all__:
        assert hasattr(module, name), f"{module.__name__}.{name} is not exported"
