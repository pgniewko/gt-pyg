"""Tests for GNM (Gaussian Network Model) encodings."""

import warnings

import numpy as np
from numpy.linalg import pinv

from gt_pyg.data import get_gnm_encodings


def test_gnm_matches_pinv_diagonal():
    """Eigendecomposition diagonal matches full pseudoinverse diagonal."""
    # Simple adjacency: path graph 0-1-2
    adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=float)

    # Reference: full pinv diagonal
    degree = np.diag(adj.sum(axis=1))
    kirchhoff = degree - adj
    expected = np.diag(pinv(kirchhoff))

    result = get_gnm_encodings(adj)

    assert result.shape == (3,)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_gnm_symmetric_molecule():
    """Symmetric graph (cycle) should give equal values for all nodes."""
    # 4-node cycle: 0-1-2-3-0
    adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ], dtype=float)

    result = get_gnm_encodings(adj)

    assert result.shape == (4,)
    np.testing.assert_allclose(result, result[0], atol=1e-12)


def test_gnm_single_atom_no_warning():
    """Single-atom molecule (1x1 zero adjacency) should not produce warnings."""
    adj = np.array([[0]], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Convert warnings to errors
        result = get_gnm_encodings(adj)

    assert result.shape == (1,)
    assert result[0] == 0.0


def test_gnm_single_atom_returns_zero():
    """Single-atom GNM diagonal should be zero (no connectivity)."""
    adj = np.array([[0]], dtype=float)
    result = get_gnm_encodings(adj)

    np.testing.assert_allclose(result, [0.0], atol=1e-12)
