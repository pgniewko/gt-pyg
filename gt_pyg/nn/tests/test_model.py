"""Tests for GraphTransformerNet freeze/unfreeze and checkpoint functionality."""

import logging

import pytest
import torch
from torch import nn

import gt_pyg
from gt_pyg.nn import GraphTransformerNet, get_checkpoint_info, load_checkpoint


@pytest.fixture
def model():
    """Create a small model for testing."""
    return GraphTransformerNet(
        node_dim_in=16,
        edge_dim_in=8,
        hidden_dim=32,
        num_gt_layers=2,
        num_heads=4,
        norm="bn",  # Use BatchNorm to test eval mode handling
    )


@pytest.fixture
def sample_input():
    """Create sample graph input."""
    return {
        "x": torch.randn(10, 16),
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        "edge_attr": torch.randn(4, 8),
        "batch": torch.zeros(10, dtype=torch.long),
    }


# ---- Freeze/Unfreeze Tests ----

def test_freeze_all(model):
    """Freezing all sets requires_grad=False for all params."""
    model.freeze()
    for p in model.parameters():
        assert not p.requires_grad


def test_freeze_with_exclude(model):
    """Excluded components remain trainable."""
    model.freeze("all", exclude="heads")

    # Heads should be trainable
    for p in model.mu_mlp.parameters():
        assert p.requires_grad
    for p in model.log_var_mlp.parameters():
        assert p.requires_grad

    # Encoder should be frozen
    for p in model.gt_layers.parameters():
        assert not p.requires_grad


def test_freeze_layer(model):
    """Freezing a specific layer only affects that layer."""
    model.freeze("gt_layer_0")

    assert not any(p.requires_grad for p in model.gt_layers[0].parameters())
    assert all(p.requires_grad for p in model.gt_layers[1].parameters())


def test_unfreeze(model):
    """Unfreezing restores requires_grad=True."""
    model.freeze()
    model.unfreeze()

    for p in model.parameters():
        assert p.requires_grad


def test_freeze_chaining(model):
    """freeze/unfreeze return self for chaining."""
    result = model.freeze("encoder").unfreeze("gt_layer_1")
    assert result is model


def test_get_frozen_status(model):
    """get_frozen_status returns correct status."""
    model.freeze("encoder")

    status = model.get_frozen_status()
    assert status["encoder"] is True
    assert status["heads"] is False


def test_get_frozen_status_unfrozen(model):
    """All components report False on a fresh model."""
    status = model.get_frozen_status()
    for name in ["embeddings", "encoder", "gt_layers", "heads"]:
        assert status[name] is False


def test_get_frozen_status_parameterless():
    """Parameterless components report None instead of vacuous True."""
    model = GraphTransformerNet(
        node_dim_in=16,
        edge_dim_in=8,
        hidden_dim=32,
        num_gt_layers=2,
        num_heads=4,
        norm="bn",
        aggregators=["sum"],  # sum pooling has no learnable params
    )
    status = model.get_frozen_status()
    assert status["pooling"] is None


def test_get_frozen_status_after_freeze_unfreeze(model):
    """Status reflects freeze then partial unfreeze."""
    model.freeze("all")
    model.unfreeze("heads")

    status = model.get_frozen_status()
    assert status["embeddings"]
    assert status["encoder"]
    assert status["gt_layers"]
    assert not status["heads"]


def test_batchnorm_eval(model):
    """BatchNorm set to eval mode when frozen."""
    model.train()
    model.freeze("encoder")

    # Input norm should be in eval mode
    assert not model.input_norm.training


def test_invalid_component(model):
    """Invalid component raises ValueError."""
    with pytest.raises(ValueError, match="Unknown component"):
        model.freeze("invalid_component")


def test_invalid_layer_index(model):
    """Invalid layer index raises ValueError."""
    with pytest.raises(ValueError, match="Invalid layer index"):
        model.freeze("gt_layer_99")


# ---- Checkpoint Tests ----

def test_checkpoint_roundtrip(model, sample_input, tmp_path):
    """Save and load produces identical outputs."""
    model.eval()
    out1, _ = model(**sample_input)

    path = tmp_path / "model.pt"
    model.save_checkpoint(path)

    model2, _ = GraphTransformerNet.load_checkpoint(path)
    model2.eval()
    out2, _ = model2(**sample_input)

    assert torch.allclose(out1, out2)


def test_checkpoint_with_optimizer(model, tmp_path):
    """Optimizer state is saved and loadable."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()  # Create state

    model.save_checkpoint(tmp_path / "model.pt", optimizer=optimizer, epoch=5)

    _, ckpt = GraphTransformerNet.load_checkpoint(tmp_path / "model.pt")

    assert "optimizer_state_dict" in ckpt
    assert ckpt["epoch"] == 5


def test_load_weights(model, sample_input, tmp_path):
    """load_weights loads into existing model."""
    model.save_checkpoint(tmp_path / "model.pt")

    model2 = GraphTransformerNet(node_dim_in=16, edge_dim_in=8, hidden_dim=32, num_gt_layers=2, num_heads=4, norm="bn")
    model2.load_weights(tmp_path / "model.pt")

    model.eval()
    model2.eval()
    out1, _ = model(**sample_input)
    out2, _ = model2(**sample_input)

    assert torch.allclose(out1, out2)


def test_get_checkpoint_info(model, tmp_path):
    """get_checkpoint_info returns metadata."""
    model.save_checkpoint(tmp_path / "model.pt", epoch=10, best_metric=0.95)

    info = get_checkpoint_info(tmp_path / "model.pt")

    assert info["epoch"] == 10
    assert info["best_metric"] == 0.95
    assert "model_config" in info
    assert "created_at" in info


def test_get_config(model):
    """get_config returns reconstructable config."""
    config = model.get_config()

    model2 = GraphTransformerNet.from_config(config)

    assert model2.hidden_dim == model.hidden_dim
    assert len(model2.gt_layers) == len(model.gt_layers)


# ---- Integration Test ----

def test_transfer_learning(model, sample_input, tmp_path):
    """Full transfer learning workflow."""
    # Save pretrained model
    model.save_checkpoint(tmp_path / "pretrained.pt")

    # Load and freeze for fine-tuning
    model2, _ = GraphTransformerNet.load_checkpoint(tmp_path / "pretrained.pt")
    model2.freeze("all", exclude="heads")

    # Verify only heads trainable
    trainable = [n for n, p in model2.named_parameters() if p.requires_grad]
    assert all("mlp" in n or "readout" in n for n in trainable)

    # Verify forward/backward works (use eval to avoid BatchNorm batch_size=1 issue)
    model2.eval()
    out, _ = model2(**sample_input)
    loss = out.sum()
    loss.backward()

    # Verify gradients only on unfrozen params
    assert model2.mu_mlp.mlp[0].weight.grad is not None
    assert model2.node_emb.weight.grad is None


# ---- Version Tests ----

def test_version_is_defined():
    """gt_pyg.__version__ is a non-empty string."""
    assert isinstance(gt_pyg.__version__, str)
    assert gt_pyg.__version__ != ""


def test_checkpoint_saves_version(model, tmp_path):
    """Checkpoint contains gt_pyg_version matching current version."""
    path = tmp_path / "model.pt"
    model.save_checkpoint(path)

    raw = torch.load(path, map_location="cpu", weights_only=False)
    assert "gt_pyg_version" in raw
    assert raw["gt_pyg_version"] == gt_pyg.__version__


def test_checkpoint_version_mismatch_warning(model, tmp_path, caplog):
    """Loading a checkpoint from a different version logs a warning."""
    path = tmp_path / "model.pt"
    model.save_checkpoint(path)

    # Tamper with the saved version
    raw = torch.load(path, map_location="cpu", weights_only=False)
    raw["gt_pyg_version"] = "0.0.0"
    torch.save(raw, path)

    with caplog.at_level(logging.WARNING, logger="gt_pyg.nn.checkpoint"):
        load_checkpoint(path)

    assert any("was saved with gt-pyg 0.0.0" in msg for msg in caplog.messages)


def test_checkpoint_missing_version_warning(model, tmp_path, caplog):
    """Loading an old checkpoint with no version field logs a warning."""
    path = tmp_path / "model.pt"
    model.save_checkpoint(path)

    # Remove the version field to simulate an old checkpoint
    raw = torch.load(path, map_location="cpu", weights_only=False)
    del raw["gt_pyg_version"]
    torch.save(raw, path)

    with caplog.at_level(logging.WARNING, logger="gt_pyg.nn.checkpoint"):
        load_checkpoint(path)

    assert any("no gt_pyg_version field" in msg for msg in caplog.messages)


def test_checkpoint_info_includes_version(model, tmp_path):
    """get_checkpoint_info returns gt_pyg_version."""
    path = tmp_path / "model.pt"
    model.save_checkpoint(path)

    info = get_checkpoint_info(path)
    assert "gt_pyg_version" in info
    assert info["gt_pyg_version"] == gt_pyg.__version__
