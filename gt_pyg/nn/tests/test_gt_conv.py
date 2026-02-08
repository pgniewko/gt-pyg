"""Tests for GTConv normalization symmetry (GitHub #40)."""

import torch
import pytest

from gt_pyg.nn.gt_conv import GTConv


class TestNormalizationSymmetry:
    """Verify the edge path uses pre-norm (matching the node path), not post-norm."""

    @pytest.fixture
    def conv(self):
        """Small GTConv with edge features and LayerNorm."""
        return GTConv(
            node_in_dim=16,
            hidden_dim=16,
            edge_in_dim=8,
            num_heads=4,
            norm="ln",
            dropout=0.0,
        )

    @pytest.fixture
    def graph(self):
        """Minimal graph: 4 nodes, 4 directed edges."""
        torch.manual_seed(42)
        x = torch.randn(4, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        edge_attr = torch.randn(4, 8)
        return x, edge_index, edge_attr

    def test_edge_output_is_not_post_normed(self, conv, graph):
        """Edge output should NOT be wrapped in an extra norm layer.

        If the edge path had a trailing norm (post-norm), then
        ``norm2e(edge_out)`` would be close to a *double* normalization of the
        raw residual.  After removing the trailing norm the output is the raw
        residual, so applying ``norm2e`` once should produce something
        *different* from the output itself.
        """
        conv.eval()
        x, edge_index, edge_attr = graph

        with torch.no_grad():
            _, edge_out = conv(x, edge_index, edge_attr)

        # If the output were already norm2e(something), then applying norm2e
        # again would roughly be a no-op only when the input is already
        # normalized.  Instead we check that norm2e(edge_out) != edge_out,
        # confirming no trailing norm was applied.
        normed = conv.norm2e(edge_out)
        assert not torch.allclose(edge_out, normed, atol=1e-5), (
            "edge_out appears to already be post-normed (norm2e is a near no-op)"
        )

    def test_norm2e_still_registered(self, conv):
        """norm2e must remain in the state dict for checkpoint compatibility."""
        state_keys = set(conv.state_dict().keys())
        assert any(k.startswith("norm2e") for k in state_keys), (
            "norm2e should still be registered for backward-compatible checkpoint loading"
        )

    def test_node_and_edge_paths_both_prenorm(self, conv, graph):
        """Both paths should follow: residual -> norm -> sublayer -> residual."""
        conv.eval()
        x, edge_index, edge_attr = graph

        with torch.no_grad():
            x_out, edge_out = conv(x, edge_index, edge_attr)

        # Neither output should be zero-mean / unit-variance across features,
        # which would be a hallmark of a trailing LayerNorm.  A residual sum
        # (pre-norm) produces outputs with non-trivial mean and variance.
        edge_mean = edge_out.mean(dim=-1)
        edge_std = edge_out.std(dim=-1)
        node_mean = x_out.mean(dim=-1)
        node_std = x_out.std(dim=-1)

        # With pre-norm the mean should generally NOT be near zero for all
        # samples, and the std should NOT be near one for all samples.
        # We check that at least one sample deviates noticeably.
        assert not (
            torch.allclose(edge_mean, torch.zeros_like(edge_mean), atol=0.1)
            and torch.allclose(edge_std, torch.ones_like(edge_std), atol=0.1)
        ), "Edge output looks post-normed (zero mean, unit std)"

        # Sanity: node path should also not look post-normed
        assert not (
            torch.allclose(node_mean, torch.zeros_like(node_mean), atol=0.1)
            and torch.allclose(node_std, torch.ones_like(node_std), atol=0.1)
        ), "Node output looks post-normed (zero mean, unit std)"
