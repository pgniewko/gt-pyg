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
# TestEdgeAttrValidation — #38
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
# TestNoMutableState — #1, #35
# ---------------------------------------------------------------------------

class TestNoMutableState:
    """GTConv must not store mutable per-forward state like _eij."""

    def test_no_eij_after_init(self, conv):
        assert not hasattr(conv, "_eij")

    def test_no_eij_after_forward(self, conv, edge_index):
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        conv(x, edge_index, edge_attr)

        assert not hasattr(conv, "_eij")

    def test_no_eij_after_forward_no_edges(self, conv_no_edge, edge_index):
        x = torch.randn(4, 16)

        conv_no_edge(x, edge_index)

        assert not hasattr(conv_no_edge, "_eij")


# ---------------------------------------------------------------------------
# TestEdgeRepresentation — #34
# ---------------------------------------------------------------------------

class TestEdgeRepresentation:
    """Edge output must depend on edge_attr, not just on node features."""

    def test_edge_output_changes_with_edge_attr(self, conv, edge_index):
        """Different edge_attr should produce different edge_out."""
        conv.eval()
        x = torch.randn(4, 16)

        edge_attr_a = torch.randn(4, 8)
        edge_attr_b = torch.randn(4, 8)

        _, edge_out_a = conv(x, edge_index, edge_attr_a)
        _, edge_out_b = conv(x, edge_index, edge_attr_b)

        assert not torch.allclose(edge_out_a, edge_out_b, atol=1e-6)

    def test_identical_nodes_different_edges(self, edge_index):
        """With identical node features, edge outputs should still differ
        when edge_attr differs — proving edge features contribute."""
        conv = GTConv(
            node_in_dim=16,
            hidden_dim=32,
            edge_in_dim=8,
            num_heads=4,
            dropout=0.0,
        )
        conv.eval()

        # All nodes identical
        x = torch.ones(4, 16)

        edge_attr_a = torch.randn(4, 8)
        edge_attr_b = torch.randn(4, 8)

        _, edge_out_a = conv(x, edge_index, edge_attr_a)
        _, edge_out_b = conv(x, edge_index, edge_attr_b)

        assert not torch.allclose(edge_out_a, edge_out_b, atol=1e-6)


# ---------------------------------------------------------------------------
# TestGradientFlow — #34
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

    def test_grad_flows_to_edge_attr(self, conv, edge_index):
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8, requires_grad=True)

        _, edge_out = conv(x, edge_index, edge_attr)
        loss = edge_out.sum()
        loss.backward()

        assert edge_attr.grad is not None
        assert edge_attr.grad.abs().sum() > 0

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

    @pytest.fixture
    def gated_conv(self):
        return GTConv(
            node_in_dim=16,
            hidden_dim=32,
            edge_in_dim=8,
            num_heads=4,
            gate=True,
            dropout=0.0,
        )

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
        conv = GTConv(
            node_in_dim=16, hidden_dim=32, edge_in_dim=8,
            num_heads=4, aggregators=["sum", "mean"], dropout=0.0,
        )
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        x_out, edge_out = conv(x, edge_index, edge_attr)
        assert x_out.shape == (4, 16)
        assert edge_out.shape == (4, 8)

    def test_dropout_nonzero(self, edge_index):
        conv = GTConv(
            node_in_dim=16, hidden_dim=32, edge_in_dim=8,
            num_heads=4, dropout=0.5,
        )
        x = torch.randn(4, 16)
        edge_attr = torch.randn(4, 8)

        # Should not error in train mode
        conv.train()
        x_out, edge_out = conv(x, edge_index, edge_attr)
        assert x_out.shape == (4, 16)

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
