"""Tests for data/utils.py: label shapes, batching, and input validation."""

import pytest
import torch
from torch_geometric.data import Batch

from gt_pyg.data.utils import get_tensor_data

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

    def test_single_nan_label(self):
        data_list = get_tensor_data([ETHANOL], [float("nan")])
        assert data_list[0].y.shape == (1, 1)
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
        expected_mask = torch.tensor([[1.0, 0.0, 1.0]])
        assert torch.equal(data_list[0].y_mask, expected_mask)


# ---------------------------------------------------------------------------
# Batching produces [B, T]
# ---------------------------------------------------------------------------

class TestBatchingShape:
    """After PyG batching, y should be [B, T] matching model output."""

    def test_batch_single_task(self):
        data_list = get_tensor_data([ETHANOL, METHANE, BENZENE], [1.0, 2.0, 3.0])
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


# ---------------------------------------------------------------------------
# Loss compatibility: no silent broadcasting
# ---------------------------------------------------------------------------

class TestLossCompatibility:
    """MSELoss should produce a scalar, not an expanded matrix."""

    def test_mse_loss_single_task(self):
        data_list = get_tensor_data([ETHANOL, METHANE], [1.0, 2.0])
        batch = Batch.from_data_list(data_list)
        pred = torch.randn(2, 1)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        assert loss.shape == ()


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

    def test_both_empty_succeeds(self):
        assert get_tensor_data([], []) == []

    def test_equal_length_succeeds(self):
        data_list = get_tensor_data([ETHANOL, METHANE], [1.0, 2.0])
        assert len(data_list) == 2


# ---------------------------------------------------------------------------
# Label-free inference (y=None)
# ---------------------------------------------------------------------------

class TestNoLabels:
    """When y is omitted, Data objects should lack y/y_mask but have features."""

    def test_no_labels_missing_y(self):
        data_list = get_tensor_data([ETHANOL])
        d = data_list[0]
        assert not hasattr(d, "y") or d.y is None
        assert not hasattr(d, "y_mask") or d.y_mask is None

    def test_no_labels_has_features(self):
        data_list = get_tensor_data([ETHANOL])
        d = data_list[0]
        assert d.x is not None and d.x.shape[0] > 0
        assert d.edge_index is not None and d.edge_index.shape[1] > 0
        assert d.edge_attr is not None and d.edge_attr.shape[0] > 0

    def test_no_labels_batch(self):
        data_list = get_tensor_data([ETHANOL, METHANE, BENZENE])
        batch = Batch.from_data_list(data_list)
        # batch.x should combine all nodes
        total_nodes = sum(d.x.shape[0] for d in data_list)
        assert batch.x.shape[0] == total_nodes
        # y should not be present
        assert not hasattr(batch, "y") or batch.y is None

    def test_with_labels_unchanged(self):
        """Existing behavior: labels produce y and y_mask."""
        data_list = get_tensor_data([ETHANOL], [[1.0, 2.0]])
        d = data_list[0]
        assert d.y is not None
        assert d.y.shape == (1, 2)
        assert d.y_mask is not None
        assert d.y_mask.shape == (1, 2)

    def test_empty_smiles_no_labels(self):
        assert get_tensor_data([]) == []
