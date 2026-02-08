"""Tests for data/utils.py: label shapes, batching, and input validation."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from gt_pyg.data.utils import get_tensor_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ETHANOL = "CCO"
METHANE = "C"
BENZENE = "c1ccccc1"


# ---------------------------------------------------------------------------
# Label shape: single-task
# ---------------------------------------------------------------------------

class TestSingleTaskLabelShape:
    """y and y_mask should have shape [1, 1] for single-task labels."""

    def test_single_float_label(self):
        data_list = get_tensor_data([ETHANOL], [1.5])
        assert data_list[0].y.shape == (1, 1)
        assert data_list[0].y_mask.shape == (1, 1)

    def test_single_int_label(self):
        data_list = get_tensor_data([ETHANOL], [3])
        assert data_list[0].y.shape == (1, 1)

    def test_single_nan_label(self):
        data_list = get_tensor_data([ETHANOL], [float("nan")])
        assert data_list[0].y.shape == (1, 1)
        assert data_list[0].y_mask.shape == (1, 1)
        assert data_list[0].y_mask[0, 0].item() == 0.0


# ---------------------------------------------------------------------------
# Label shape: multi-task
# ---------------------------------------------------------------------------

class TestMultitaskLabelShape:
    """y and y_mask should have shape [1, T] for multi-task labels."""

    def test_two_task_labels(self):
        data_list = get_tensor_data([ETHANOL], [[1.0, 2.0]])
        assert data_list[0].y.shape == (1, 2)
        assert data_list[0].y_mask.shape == (1, 2)

    def test_three_task_with_nan(self):
        data_list = get_tensor_data([ETHANOL], [[1.0, None, 3.0]])
        assert data_list[0].y.shape == (1, 3)
        assert data_list[0].y_mask.shape == (1, 3)
        expected_mask = torch.tensor([[1.0, 0.0, 1.0]])
        assert torch.equal(data_list[0].y_mask, expected_mask)


# ---------------------------------------------------------------------------
# Batching produces [B, T]
# ---------------------------------------------------------------------------

class TestBatchingShape:
    """After PyG batching, y should be [B, T] matching model output."""

    def test_batch_single_task(self):
        data_list = get_tensor_data(
            [ETHANOL, METHANE, BENZENE],
            [1.0, 2.0, 3.0],
        )
        batch = Batch.from_data_list(data_list)
        assert batch.y.shape == (3, 1)
        assert batch.y_mask.shape == (3, 1)

    def test_batch_multitask(self):
        data_list = get_tensor_data(
            [ETHANOL, METHANE],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
        batch = Batch.from_data_list(data_list)
        assert batch.y.shape == (2, 3)
        assert batch.y_mask.shape == (2, 3)


# ---------------------------------------------------------------------------
# Loss compatibility: no silent broadcasting
# ---------------------------------------------------------------------------

class TestLossCompatibility:
    """MSELoss should produce a scalar (not an expanded matrix)."""

    def test_mse_loss_single_task(self):
        data_list = get_tensor_data(
            [ETHANOL, METHANE],
            [1.0, 2.0],
        )
        batch = Batch.from_data_list(data_list)
        # Simulate model output shape [B, 1]
        pred = torch.randn(2, 1)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        assert loss.shape == ()  # scalar

    def test_mse_loss_multitask(self):
        data_list = get_tensor_data(
            [ETHANOL, METHANE],
            [[1.0, 2.0], [3.0, 4.0]],
        )
        batch = Batch.from_data_list(data_list)
        pred = torch.randn(2, 2)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        assert loss.shape == ()  # scalar


# ---------------------------------------------------------------------------
# Input validation: length mismatch
# ---------------------------------------------------------------------------

class TestLengthValidation:
    """get_tensor_data should raise ValueError on mismatched lengths."""

    def test_more_smiles_than_labels(self):
        with pytest.raises(ValueError, match="same length"):
            get_tensor_data([ETHANOL, METHANE, BENZENE], [1.0, 2.0])

    def test_more_labels_than_smiles(self):
        with pytest.raises(ValueError, match="same length"):
            get_tensor_data([ETHANOL], [1.0, 2.0])

    def test_empty_smiles_nonempty_labels(self):
        with pytest.raises(ValueError, match="same length"):
            get_tensor_data([], [1.0])

    def test_nonempty_smiles_empty_labels(self):
        with pytest.raises(ValueError, match="same length"):
            get_tensor_data([ETHANOL], [])

    def test_both_empty_succeeds(self):
        result = get_tensor_data([], [])
        assert result == []

    def test_equal_length_succeeds(self):
        data_list = get_tensor_data([ETHANOL, METHANE], [1.0, 2.0])
        assert len(data_list) == 2


# ---------------------------------------------------------------------------
# Existing behavior: valid SMILES processing
# ---------------------------------------------------------------------------

class TestBasicBehavior:
    """Basic sanity checks that existing behavior is preserved."""

    def test_returns_data_objects(self):
        from torch_geometric.data import Data
        data_list = get_tensor_data([ETHANOL], [1.0])
        assert len(data_list) == 1
        assert isinstance(data_list[0], Data)

    def test_node_features_present(self):
        data_list = get_tensor_data([ETHANOL], [1.0])
        assert data_list[0].x is not None
        assert data_list[0].x.dim() == 2  # [N, F]

    def test_edge_features_present(self):
        data_list = get_tensor_data([ETHANOL], [1.0])
        assert data_list[0].edge_index is not None
        assert data_list[0].edge_attr is not None

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="Failed to canonicalize"):
            get_tensor_data(["not_a_smiles"], [1.0])
