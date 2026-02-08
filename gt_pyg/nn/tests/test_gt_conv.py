"""Tests for the GTConv (Graph Transformer Convolution) layer."""

import pytest
import torch

from gt_pyg.nn.gt_conv import GTConv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def edge_index():
    """Simple 4-node cycle graph: 0->1->2->3->0."""
    return torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])


@pytest.fixture
def conv():
    """GTConv with edge features, LayerNorm, no gate."""
    return GTConv(
        node_in_dim=16,
        hidden_dim=32,
        edge_in_dim=8,
        num_heads=4,
        dropout=0.0,
    )


@pytest.fixture
def conv_no_edge():
    """GTConv without edge features."""
    return GTConv(
        node_in_dim=16,
        hidden_dim=32,
        edge_in_dim=None,
        num_heads=4,
        dropout=0.0,
    )


@pytest.fixture
def gated_conv():
    """GTConv with gating enabled."""
    return GTConv(
        node_in_dim=16,
        hidden_dim=32,
        edge_in_dim=8,
        num_heads=4,
        gate=True,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# TestForwardPass — shapes and return types
# ---------------------------------------------------------------------------

class TestForwardPass:
    """Basic forward-pass sanity checks."""

    def test_output_shapes_with_edges(self, conv, edge_index):
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        x_out, edge_out = conv(x, edge_index, edge_attr)

        assert x_out.shape == (4, 16)
        assert edge_out.shape == (4, 8)

    def test_output_shapes_without_edges(self, conv_no_edge, edge_index):
        x = torch.randn(4, 16)

        x_out, edge_out = conv_no_edge(x, edge_index)

        assert x_out.shape == (4, 16)
        assert edge_out is None

    def test_return_is_tuple(self, conv, edge_index):
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        result = conv(x, edge_index, edge_attr)

        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TestEdgeAttrValidation
# ---------------------------------------------------------------------------

class TestEdgeAttrValidation:
    """edge_attr=None should raise when edge_in_dim was set."""

    def test_raises_when_edge_attr_missing(self, conv, edge_index):
        x = torch.randn(4, 16)

        with pytest.raises(ValueError, match="edge_in_dim was set"):
            conv(x, edge_index, edge_attr=None)

    def test_no_raise_when_edge_in_dim_none(self, conv_no_edge, edge_index):
        x = torch.randn(4, 16)

        # Should not raise
        x_out, edge_out = conv_no_edge(x, edge_index, edge_attr=None)
        assert x_out is not None


# ---------------------------------------------------------------------------
# TestEdgeRepresentation
# ---------------------------------------------------------------------------

class TestEdgeRepresentation:
    """Edge output must depend on edge_attr, not just on node features."""

    def test_edge_output_changes_with_edge_attr(self, conv, edge_index):
        """Different edge_attr should produce different edge_out."""
        torch.manual_seed(42)
        conv.eval()
        x = torch.randn(4, 16)

        edge_attr_a = torch.randn(4, 8)
        edge_attr_b = torch.randn(4, 8)

        _, edge_out_a = conv(x, edge_index, edge_attr_a)
        _, edge_out_b = conv(x, edge_index, edge_attr_b)

        assert not torch.allclose(edge_out_a, edge_out_b, atol=1e-6)


# ---------------------------------------------------------------------------
# TestGradientFlow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Gradients must reach both x and edge_attr."""

    def test_grad_flows_to_x(self, conv, edge_index):
        x = torch.randn(4, 16, requires_grad=True)
        edge_attr = torch.randn(4, 8)

        x_out, _ = conv(x, edge_index, edge_attr)
        loss = x_out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_edge_grad_through_edge_update(self, conv, edge_index):
        """edge_attr gradient flows specifically through the edge update path,
        not just through the attention bias/values in the node path."""
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8, requires_grad=True)

        _, edge_out = conv(x, edge_index, edge_attr)

        # Only backprop through the edge output
        loss = edge_out.sum()
        loss.backward()

        # WE_value should receive gradient from the edge update path
        assert conv.WE_value.weight.grad is not None
        assert conv.WE_value.weight.grad.abs().sum() > 0

        # WOe should receive gradient from the edge update path
        assert conv.WOe.weight.grad is not None
        assert conv.WOe.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# TestGating
# ---------------------------------------------------------------------------

class TestGating:
    """Tests for the optional gating mechanism."""

    def test_gated_forward(self, gated_conv, edge_index):
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        x_out, edge_out = gated_conv(x, edge_index, edge_attr)

        assert x_out.shape == (4, 16)
        assert edge_out.shape == (4, 8)

    def test_gated_gradient_flow(self, gated_conv, edge_index):
        x = torch.randn(4, 16, requires_grad=True)
        edge_attr = torch.randn(4, 8, requires_grad=True)

        x_out, edge_out = gated_conv(x, edge_index, edge_attr)
        loss = x_out.sum() + edge_out.sum()
        loss.backward()

        assert x.grad is not None
        assert edge_attr.grad is not None

    def test_gated_vs_ungated_differ(self, edge_index):
        """Gated and ungated outputs should differ (different param init aside,
        the gate path itself introduces sigmoid gating)."""
        torch.manual_seed(42)
        ungated = GTConv(
            node_in_dim=16, hidden_dim=32, edge_in_dim=8,
            num_heads=4, gate=False, dropout=0.0,
        )
        torch.manual_seed(42)
        gated = GTConv(
            node_in_dim=16, hidden_dim=32, edge_in_dim=8,
            num_heads=4, gate=True, dropout=0.0,
        )

        ungated.eval()
        gated.eval()

        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        x_ung, _ = ungated(x, edge_index, edge_attr)
        x_gat, _ = gated(x, edge_index, edge_attr)

        # They share Q/K/V/O weights but gated has extra gate params
        assert not torch.allclose(x_ung, x_gat, atol=1e-6)


# ---------------------------------------------------------------------------
# TestConfiguration
# ---------------------------------------------------------------------------

class TestConfiguration:
    """Tests for various constructor configurations."""

    def test_batchnorm(self, edge_index):
        conv = GTConv(
            node_in_dim=16, hidden_dim=32, edge_in_dim=8,
            num_heads=4, norm="bn", dropout=0.0,
        )
        # BN needs > 1 sample
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        x_out, edge_out = conv(x, edge_index, edge_attr)
        assert x_out.shape == (4, 16)
        assert edge_out.shape == (4, 8)
        # Output should differ from input (non-trivial transform)
        assert not torch.allclose(x_out, x, atol=1e-6)

    def test_qkv_bias(self, edge_index):
        conv = GTConv(
            node_in_dim=16, hidden_dim=32, edge_in_dim=8,
            num_heads=4, qkv_bias=True, dropout=0.0,
        )
        assert conv.WQ.bias is not None

        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)
        x_out, _ = conv(x, edge_index, edge_attr)
        assert x_out.shape == (4, 16)

    def test_multi_aggregator(self, edge_index):
        torch.manual_seed(99)
        conv_single = GTConv(
            node_in_dim=16, hidden_dim=32, edge_in_dim=8,
            num_heads=4, aggregators=["sum"], dropout=0.0,
        )
        torch.manual_seed(99)
        conv_multi = GTConv(
            node_in_dim=16, hidden_dim=32, edge_in_dim=8,
            num_heads=4, aggregators=["sum", "mean"], dropout=0.0,
        )
        conv_single.eval()
        conv_multi.eval()

        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        x_out_s, _ = conv_single(x, edge_index, edge_attr)
        x_out_m, _ = conv_multi(x, edge_index, edge_attr)

        # Multi-aggregator should produce different outputs
        assert x_out_s.shape == x_out_m.shape
        assert not torch.allclose(x_out_s, x_out_m, atol=1e-6)

    def test_dropout_nonzero(self, edge_index):
        conv = GTConv(
            node_in_dim=16, hidden_dim=32, edge_in_dim=8,
            num_heads=4, dropout=0.5,
        )
        torch.manual_seed(0)
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        # Train mode: dropout active
        conv.train()
        torch.manual_seed(1)
        x_train, _ = conv(x, edge_index, edge_attr)

        # Eval mode: dropout inactive
        conv.eval()
        x_eval, _ = conv(x, edge_index, edge_attr)

        # Outputs should differ because dropout is active in train mode
        assert not torch.allclose(x_train, x_eval, atol=1e-6)

    def test_invalid_hidden_dim(self):
        with pytest.raises(ValueError, match="divisible by num_heads"):
            GTConv(node_in_dim=16, hidden_dim=31, num_heads=4)

    def test_invalid_edge_in_dim(self):
        with pytest.raises(ValueError, match="edge_in_dim must be positive"):
            GTConv(node_in_dim=16, hidden_dim=32, edge_in_dim=0, num_heads=4)

    def test_default_dropout_is_point_one(self):
        """Default dropout should be 0.1 (synced with GraphTransformerNet)."""
        conv = GTConv(node_in_dim=16, hidden_dim=32, num_heads=4)
        assert conv.dropout_p == 0.1


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Verify GTConv rejects invalid constructor arguments with clear errors."""

    def test_num_heads_zero_raises(self):
        """num_heads=0 should raise ValueError, not ZeroDivisionError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            GTConv(node_in_dim=16, hidden_dim=16, num_heads=0)

    def test_num_heads_negative_raises(self):
        """Negative num_heads should raise ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            GTConv(node_in_dim=16, hidden_dim=16, num_heads=-1)


# ---------------------------------------------------------------------------
# TestNormalizationSymmetry
# ---------------------------------------------------------------------------

class TestNormalizationSymmetry:
    """Verify the edge path uses pre-norm (matching the node path), not post-norm."""

    def test_edge_output_is_not_post_normed(self, conv, edge_index):
        """Edge output should NOT be wrapped in an extra norm layer.

        With pre-norm, the output is a raw residual sum, so it should NOT
        have zero mean and unit variance across features.
        """
        torch.manual_seed(42)
        conv.eval()
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        with torch.no_grad():
            _, edge_out = conv(x, edge_index, edge_attr)

        # A post-normed output would have ~zero mean and ~unit std.
        # Pre-norm residual output should deviate from that.
        edge_mean = edge_out.mean(dim=-1)
        edge_std = edge_out.std(dim=-1)
        assert not (
            torch.allclose(edge_mean, torch.zeros_like(edge_mean), atol=0.1)
            and torch.allclose(edge_std, torch.ones_like(edge_std), atol=0.1)
        ), "Edge output looks post-normed (zero mean, unit std)"


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Eval-mode repeated forward should be deterministic."""

    def test_deterministic_eval(self, conv, edge_index):
        conv.eval()

        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        x_out1, edge_out1 = conv(x, edge_index, edge_attr)
        x_out2, edge_out2 = conv(x, edge_index, edge_attr)

        assert torch.allclose(x_out1, x_out2, atol=1e-6)
        assert torch.allclose(edge_out1, edge_out2, atol=1e-6)

    def test_deterministic_eval_no_edges(self, conv_no_edge, edge_index):
        conv_no_edge.eval()

        x = torch.randn(4, 16)

        x_out1, _ = conv_no_edge(x, edge_index)
        x_out2, _ = conv_no_edge(x, edge_index)

        assert torch.allclose(x_out1, x_out2, atol=1e-6)
