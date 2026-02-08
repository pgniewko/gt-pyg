"""Tests for checkpoint utilities (gt_pyg.nn.checkpoint)."""

import pytest
import torch
from torch import nn

from gt_pyg.nn.checkpoint import save_checkpoint, load_checkpoint, get_checkpoint_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model for checkpoint tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def model():
    return _TinyModel()


# ---------------------------------------------------------------------------
# TestSaveLoad
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """save_checkpoint / load_checkpoint round-trip."""

    def test_roundtrip_state_dict(self, model, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(model, path)
        ckpt = load_checkpoint(path)

        model2 = _TinyModel()
        model2.load_state_dict(ckpt["model_state_dict"])

        x = torch.randn(3, 4)
        model.eval()
        model2.eval()
        assert torch.allclose(model(x), model2(x))

    def test_pt_suffix_added(self, model, tmp_path):
        path = tmp_path / "ckpt"
        save_checkpoint(model, path)
        assert (tmp_path / "ckpt.pt").exists()

    def test_creates_parent_dirs(self, model, tmp_path):
        path = tmp_path / "a" / "b" / "ckpt.pt"
        save_checkpoint(model, path)
        assert path.exists()

    def test_optional_fields_saved(self, model, tmp_path):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        save_checkpoint(
            model, tmp_path / "ckpt.pt",
            config={"dim": 4},
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=7,
            global_step=100,
            best_metric=0.99,
            extra={"note": "test"},
        )

        ckpt = load_checkpoint(tmp_path / "ckpt.pt")
        assert ckpt["model_config"] == {"dim": 4}
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert ckpt["epoch"] == 7
        assert ckpt["global_step"] == 100
        assert ckpt["best_metric"] == 0.99
        assert ckpt["extra"]["note"] == "test"

    def test_version_and_timestamp_saved(self, model, tmp_path):
        import gt_pyg

        save_checkpoint(model, tmp_path / "ckpt.pt")
        ckpt = load_checkpoint(tmp_path / "ckpt.pt")

        assert ckpt["gt_pyg_version"] == gt_pyg.__version__
        assert "created_at" in ckpt
        assert ckpt["checkpoint_version"] == 1


# ---------------------------------------------------------------------------
# TestGetCheckpointInfo
# ---------------------------------------------------------------------------

class TestGetCheckpointInfo:
    """get_checkpoint_info metadata extraction."""

    def test_returns_metadata_keys(self, model, tmp_path):
        save_checkpoint(
            model, tmp_path / "ckpt.pt",
            config={"dim": 4},
            epoch=3,
            best_metric=0.5,
        )

        info = get_checkpoint_info(tmp_path / "ckpt.pt")
        assert info["epoch"] == 3
        assert info["best_metric"] == 0.5
        assert info["model_config"] == {"dim": 4}
        assert "gt_pyg_version" in info
        assert "created_at" in info

    def test_excludes_state_dicts(self, model, tmp_path):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(model, tmp_path / "ckpt.pt", optimizer=optimizer)

        info = get_checkpoint_info(tmp_path / "ckpt.pt")
        assert "model_state_dict" not in info
        assert "optimizer_state_dict" not in info

    def test_frozen_status_extracted_from_extra(self, model, tmp_path):
        """frozen_status is stored inside extra by GraphTransformerNet.save_checkpoint.
        get_checkpoint_info must extract it from there (bug #2 / dead code #20)."""
        frozen = {"encoder": True, "heads": False}
        save_checkpoint(
            model, tmp_path / "ckpt.pt",
            extra={"frozen_status": frozen},
        )

        info = get_checkpoint_info(tmp_path / "ckpt.pt")
        assert "frozen_status" in info
        assert info["frozen_status"] == frozen

    def test_frozen_status_missing_gracefully(self, model, tmp_path):
        """No frozen_status when extra doesn't contain it."""
        save_checkpoint(model, tmp_path / "ckpt.pt")

        info = get_checkpoint_info(tmp_path / "ckpt.pt")
        assert "frozen_status" not in info

    def test_frozen_status_with_no_extra(self, model, tmp_path):
        """No crash when extra is absent entirely."""
        save_checkpoint(model, tmp_path / "ckpt.pt")

        info = get_checkpoint_info(tmp_path / "ckpt.pt")
        assert "frozen_status" not in info
        assert "extra" not in info


# ---------------------------------------------------------------------------
# TestLoadCheckpointWarnings
# ---------------------------------------------------------------------------

class TestLoadCheckpointWarnings:
    """Version mismatch and missing version warnings."""

    def test_warns_on_version_mismatch(self, model, tmp_path, caplog):
        import logging

        save_checkpoint(model, tmp_path / "ckpt.pt")

        # Tamper version
        raw = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
        raw["gt_pyg_version"] = "0.0.0"
        torch.save(raw, tmp_path / "ckpt.pt")

        with caplog.at_level(logging.WARNING, logger="gt_pyg.nn.checkpoint"):
            load_checkpoint(tmp_path / "ckpt.pt")

        assert any("was saved with gt-pyg 0.0.0" in msg for msg in caplog.messages)

    def test_warns_on_missing_version(self, model, tmp_path, caplog):
        import logging

        save_checkpoint(model, tmp_path / "ckpt.pt")

        raw = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
        del raw["gt_pyg_version"]
        torch.save(raw, tmp_path / "ckpt.pt")

        with caplog.at_level(logging.WARNING, logger="gt_pyg.nn.checkpoint"):
            load_checkpoint(tmp_path / "ckpt.pt")

        assert any("no gt_pyg_version field" in msg for msg in caplog.messages)

    def test_no_warning_on_matching_version(self, model, tmp_path, caplog):
        import logging

        save_checkpoint(model, tmp_path / "ckpt.pt")

        with caplog.at_level(logging.WARNING, logger="gt_pyg.nn.checkpoint"):
            load_checkpoint(tmp_path / "ckpt.pt")

        assert not any("gt-pyg" in msg.lower() or "gt_pyg_version" in msg for msg in caplog.messages)
