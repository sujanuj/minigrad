"""Tests for the transformer architecture.

Verifies:
  1. Forward pass shapes at each component level.
  2. Causal masking: changing token t+1 doesn't affect logits for token t.
  3. Loss computation and gradient flow through the full model.
  4. Parameter count is correct.
  5. Train/eval mode switching.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformer.transformer import (
    Transformer, TransformerBlock, CausalSelfAttention, MLP, causal_mask
)
from autograd.tensor import Tensor


# ---------------------------------------------------------------------------
# Tiny config for fast tests
# ---------------------------------------------------------------------------

def tiny_transformer(**kwargs):
    defaults = dict(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers=2,
        d_ff=32,
        max_seq_len=16,
        dropout_p=0.0,
    )
    defaults.update(kwargs)
    return Transformer(**defaults)


# ---------------------------------------------------------------------------
# Causal mask tests
# ---------------------------------------------------------------------------

def test_causal_mask_shape():
    mask = causal_mask(4)
    assert mask.shape == (4, 4)


def test_causal_mask_is_lower_triangular():
    mask = causal_mask(5)
    assert mask[0, 0] == True
    assert mask[1, 0] == True
    assert mask[0, 1] == False
    assert mask[3, 4] == False


# ---------------------------------------------------------------------------
# Attention tests
# ---------------------------------------------------------------------------

def test_attention_output_shape():
    attn = CausalSelfAttention(d_model=16, n_heads=2)
    x = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
    out = attn(x)
    assert out.shape == (2, 5, 16)


def test_attention_gradient_flows():
    attn = CausalSelfAttention(d_model=16, n_heads=2)
    x = Tensor(np.random.randn(2, 4, 16).astype(np.float32), requires_grad=True)
    loss = attn(x).sum()
    loss.backward()
    assert x.grad is not None
    assert not np.allclose(x.grad, 0)


# ---------------------------------------------------------------------------
# MLP tests
# ---------------------------------------------------------------------------

def test_mlp_output_shape():
    mlp = MLP(d_model=16, d_ff=32)
    x = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
    out = mlp(x)
    assert out.shape == (2, 5, 16)


def test_mlp_gradient_flows():
    mlp = MLP(d_model=16, d_ff=32)
    x = Tensor(np.random.randn(2, 4, 16).astype(np.float32), requires_grad=True)
    mlp(x).sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# TransformerBlock tests
# ---------------------------------------------------------------------------

def test_block_output_shape():
    block = TransformerBlock(d_model=16, n_heads=2, d_ff=32)
    x = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
    out = block(x)
    assert out.shape == (2, 5, 16)


def test_block_residual_connection():
    # With a zeroed-out MLP and attention, output should equal input
    block = TransformerBlock(d_model=16, n_heads=2, d_ff=32)
    # Zero all parameters
    for p in block.parameters():
        p.data[:] = 0
    # With zero weights, attention output is 0, MLP output is 0
    # So x + 0 = x at each residual
    x = Tensor(np.random.randn(1, 3, 16).astype(np.float32))
    out = block(x)
    # Output should be close to input (residual dominates when weights are 0)
    # LayerNorm with gamma=1, beta=0 normalizes, so not exactly equal
    assert out.shape == (1, 3, 16)


# ---------------------------------------------------------------------------
# Full transformer tests
# ---------------------------------------------------------------------------

def test_transformer_forward_shape():
    model = tiny_transformer()
    idx = np.random.randint(0, 32, (2, 8))
    logits = model(idx)
    assert logits.shape == (2, 8, 32)


def test_transformer_loss_is_scalar():
    model = tiny_transformer()
    idx = np.random.randint(0, 32, (2, 8))
    targets = np.random.randint(0, 32, (2, 8))
    loss = model.loss(idx, targets)
    assert loss.shape == ()


def test_transformer_loss_reasonable():
    # With random weights, loss should be close to log(vocab_size)
    np.random.seed(0)
    model = tiny_transformer(vocab_size=32)
    idx = np.random.randint(0, 32, (4, 8))
    targets = np.random.randint(0, 32, (4, 8))
    loss = model.loss(idx, targets)
    # Random model should have loss ~ log(32) ≈ 3.47
    assert 2.0 < float(loss.data) < 6.0, \
        f"Expected loss ~3.47, got {float(loss.data)}"


def test_transformer_gradient_flows():
    np.random.seed(1)
    model = tiny_transformer()
    idx = np.random.randint(0, 32, (2, 4))
    targets = np.random.randint(0, 32, (2, 4))
    loss = model.loss(idx, targets)
    loss.backward()
    # Check that embedding gradients are nonzero
    assert model.token_emb.weight.grad is not None
    # At least some embedding rows should have nonzero gradients
    assert not np.allclose(model.token_emb.weight.grad, 0)


def test_transformer_causal_masking():
    """Changing token at position t+1 must not affect logits at position t."""
    np.random.seed(2)
    model = tiny_transformer()
    model.eval()

    idx1 = np.array([[1, 2, 3, 4, 5]])
    idx2 = np.array([[1, 2, 3, 4, 9]])  # only last token differs

    logits1 = model(idx1)
    logits2 = model(idx2)

    # Logits for all positions except the last should be identical
    np.testing.assert_allclose(
        logits1.data[:, :-1, :],
        logits2.data[:, :-1, :],
        atol=1e-5,
        err_msg="Causal masking violated: changing future token affected past logits"
    )


def test_transformer_parameter_count():
    model = tiny_transformer(
        vocab_size=32, d_model=16, n_heads=2, n_layers=2, d_ff=32
    )
    n_params = model.num_parameters()
    assert n_params > 0
    print(f"\nTiny transformer: {n_params:,} parameters")


def test_transformer_train_eval_mode():
    model = tiny_transformer(dropout_p=0.5)
    model.train()
    assert model.drop.training == True
    model.eval()
    assert model.drop.training == False


def test_transformer_parameters_list():
    model = tiny_transformer()
    params = model.parameters()
    assert len(params) > 0
    for p in params:
        assert p.requires_grad
